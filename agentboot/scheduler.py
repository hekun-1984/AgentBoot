"""并行批次调度器。

1. 按拓扑序将步骤分成批次
2. 同一批次内没有依赖关系的步骤并行执行
3. 任一步骤失败则中止后续批次
4. 记录完整执行轨迹
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agentboot.exceptions import (
    ExecutionError,
    PlanValidationError,
    RegistrationError,
    RoutingError,
)
from agentboot.logger import get_logger
from agentboot.models import ExecutionPlan, ExecutionResult, Step, StepStatus
from agentboot.registry import Registry
from agentboot.router import Router
from agentboot.utils.tracing import ExecutionTracer

logger = get_logger(__name__)


def hydrate_plan(plan: ExecutionPlan, registry: Registry) -> ExecutionPlan:
    """用子图/能力上的声明补全计划里留空的依赖、并行组和白名单。"""
    for step in plan.steps:
        component = registry.get_subgraph(step.id) or registry.get_skill(step.id)
        if component is None:
            continue
        meta = component()
        declared = list(getattr(meta, "depends_on", []) or [])
        if not step.depends_on and declared:
            step.depends_on = declared
        if step.parallel_group is None and getattr(meta, "parallel_group", None):
            step.parallel_group = meta.parallel_group
        allowed = getattr(meta, "allowed_skills", None)
        if allowed and "allowed_skills" not in step.config:
            step.config["allowed_skills"] = list(allowed)
    return plan


class Scheduler:
    """并行批次调度器。"""

    def __init__(self, registry: Registry, router: Router, llm_model: str = "default") -> None:
        self.registry = registry
        self.router = router
        self.llm_model = llm_model
        self.llm_client: Any | None = None

    async def start(self) -> None:
        """调度器就绪。组件已注册后即可接受计划。"""
        logger.info(
            "Scheduler ready. agents=%s subgraphs=%s skills=%s",
            self.registry.list_agents(),
            self.registry.list_subgraphs(),
            self.registry.list_skills(),
        )

    def explain(self, plan: ExecutionPlan) -> list[list[str]]:
        hydrate_plan(plan, self.registry)
        return [[step.id for step in batch] for batch in self._group_into_batches(plan.steps)]

    async def execute(self, plan: ExecutionPlan, context: dict[str, Any] | None = None) -> ExecutionResult:
        """执行一个计划。"""
        started = time.monotonic()
        context = dict(context or {})
        hydrate_plan(plan, self.registry)
        logger.info("Executing plan: %s", plan.plan_id)
        tracer = ExecutionTracer()
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        batches = self._group_into_batches(plan.steps)
        tracer.record(
            event="plan",
            plan_id=plan.plan_id,
            source=plan.metadata.get("source"),
        )
        tracer.record(event="batches", batches=[[step.id for step in batch] for batch in batches])
        for batch_idx, batch in enumerate(batches):
            logger.info(
                "Batch %d/%d: %s",
                batch_idx + 1,
                len(batches),
                [step.id for step in batch],
            )
            snapshot = {**context, **results}
            outcomes = await self._run_batch(batch, snapshot, plan.agent)
            failed = False
            for step, outcome in outcomes:
                if isinstance(outcome, Exception):
                    errors[step.id] = str(outcome)
                    tracer.record(
                        step=step.id,
                        status=StepStatus.FAILED.value,
                        batch=batch_idx + 1,
                        error=str(outcome),
                    )
                    failed = True
                    logger.error("Step %s failed: %s", step.id, outcome)
                else:
                    results[step.id] = outcome
                    tracer.record(
                        step=step.id,
                        status=StepStatus.SUCCESS.value,
                        batch=batch_idx + 1,
                        duration_ms=outcome.pop("__duration_ms", None) if isinstance(outcome, dict) else None,
                    )
            if failed:
                for pending in plan.steps:
                    if pending.id in results or pending.id in errors:
                        continue
                    tracer.record(
                        step=pending.id,
                        status=StepStatus.SKIPPED.value,
                        reason="upstream_failed",
                    )
                break
        duration_ms = int((time.monotonic() - started) * 1000)
        status = "success" if not errors else "failed"
        return ExecutionResult(
            plan_id=plan.plan_id,
            status=status,
            results=results,
            errors=errors,
            trace=tracer.export(),
            duration_ms=duration_ms,
        )

    async def run_step(
        self,
        step: Step,
        upstream: dict[str, Any],
        agent_name: str,
    ) -> dict[str, Any]:
        """执行单个步骤。LangGraph 适配器复用同一条调用链。"""
        timeout = self._timeout_for(step, agent_name)
        started = time.monotonic()
        try:
            if timeout:
                result = await asyncio.wait_for(
                    self._invoke_step(step, upstream, agent_name),
                    timeout=timeout,
                )
            else:
                result = await self._invoke_step(step, upstream, agent_name)
        except asyncio.TimeoutError:
            raise ExecutionError(f"Step '{step.id}' timed out after {timeout}s.") from None
        if not isinstance(result, dict):
            result = {"value": result}
        result["__duration_ms"] = int((time.monotonic() - started) * 1000)
        return result

    async def _run_batch(
        self,
        batch: list[Step],
        snapshot: dict[str, Any],
        agent_name: str,
    ) -> list[tuple[Step, dict[str, Any] | Exception]]:
        tasks = [
            self.run_step(step, dict(snapshot), agent_name)
            for step in batch
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        return list(zip(batch, raw, strict=True))

    async def _invoke_step(
        self,
        step: Step,
        upstream: dict[str, Any],
        agent_name: str,
    ) -> dict[str, Any]:
        if step.type == "subgraph":
            return await self._invoke_subgraph(step, upstream, agent_name)
        if step.type == "skill":
            self.router.check_whitelist(agent_name, [step.id])
            return await self._call_skill(step.id, upstream)
        if step.type == "llm":
            return await self._invoke_llm(step, upstream)
        if step.type == "condition":
            return self._invoke_condition(step, upstream)
        raise ExecutionError(f"Unsupported step type '{step.type}'.")

    async def _invoke_subgraph(
        self,
        step: Step,
        upstream: dict[str, Any],
        agent_name: str,
    ) -> dict[str, Any]:
        allowed = step.config.get("allowed_skills")
        if allowed:
            self.router.check_whitelist(agent_name, list(allowed))
        graph_class = self.registry.get_subgraph(step.id)
        if graph_class is None:
            raise ExecutionError(f"Unknown step type or unregistered: {step.id}")
        graph = graph_class()
        graph.bind_runner(self._runner_for(step, agent_name))
        return await graph.execute(upstream)

    def _runner_for(self, step: Step, agent_name: str):
        allowed = step.config.get("allowed_skills")

        async def runner(skill_name: str, payload: dict[str, Any]) -> dict[str, Any]:
            if isinstance(allowed, list) and skill_name not in allowed:
                raise RoutingError(
                    f"SubGraph '{step.id}' cannot call skill '{skill_name}'. Declared: {allowed}."
                )
            self.router.check_whitelist(agent_name, [skill_name])
            return await self._call_skill(skill_name, payload)

        return runner

    async def _call_skill(self, skill_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        skill_class = self.registry.get_skill(skill_name)
        if skill_class is None:
            raise RegistrationError(f"Skill '{skill_name}' is not registered.")
        skill = skill_class()
        result = await skill.execute(payload)
        if not isinstance(result, dict):
            raise ExecutionError(f"Skill '{skill_name}' must return a dict.")
        return result

    async def _invoke_llm(self, step: Step, upstream: dict[str, Any]) -> dict[str, Any]:
        if self.llm_client is None:
            raise ExecutionError(
                "LLM step requires AGENTBOOT_LLM_API_KEY or an injected client."
            )
        import json

        prompt = str(step.config.get("prompt", ""))
        payload = prompt + "\n\nContext:\n" + json.dumps(upstream, ensure_ascii=False, default=str)
        text = await self.llm_client.complete(payload)
        return {"text": text, "model": self.llm_model}

    def _invoke_condition(self, step: Step, upstream: dict[str, Any]) -> dict[str, Any]:
        path = step.config.get("path")
        if not path:
            return {"passed": True}
        actual = _lookup_path(upstream, str(path))
        expected = step.config.get("value")
        operator = str(step.config.get("op", "eq"))
        comparators = {
            "eq": lambda left, right: left == right,
            "gte": lambda left, right: left >= right,
            "lte": lambda left, right: left <= right,
            "gt": lambda left, right: left > right,
            "lt": lambda left, right: left < right,
            "exists": lambda left, _right: left is not None,
        }
        compare = comparators.get(operator)
        if compare is None:
            raise ExecutionError(f"Unsupported condition op '{operator}'.")
        try:
            passed = bool(compare(actual, expected))
        except TypeError as exc:
            raise ExecutionError(f"Condition '{step.id}' cannot compare values.") from exc
        return {"passed": passed, "actual": actual}

    def _timeout_for(self, step: Step, agent_name: str) -> float | None:
        if step.timeout:
            return float(step.timeout)
        route_timeout = self.router.get_route(agent_name).get("timeout")
        if route_timeout:
            return float(route_timeout)
        return None

    def _group_into_batches(self, steps: list[Step]) -> list[list[Step]]:
        """按依赖关系分批。同一批次内的步骤可以并行执行。"""
        if not steps:
            return []
        order = {step.id: index for index, step in enumerate(steps)}
        step_map = {step.id: step for step in steps}
        dependents: dict[str, list[str]] = {step.id: [] for step in steps}
        indegree = {step.id: 0 for step in steps}
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id not in step_map:
                    raise PlanValidationError(
                        f"Step '{step.id}' depends on unknown step '{dep_id}'."
                    )
                dependents[dep_id].append(step.id)
                indegree[step.id] += 1
        ready = [step_map[step_id] for step_id, degree in indegree.items() if degree == 0]
        batches: list[list[Step]] = []
        seen = 0
        while ready:
            batch = sorted(ready, key=lambda item: order[item.id])
            batches.append(batch)
            seen += len(batch)
            next_ready: list[Step] = []
            for step in batch:
                for child_id in dependents[step.id]:
                    indegree[child_id] -= 1
                    if indegree[child_id] == 0:
                        next_ready.append(step_map[child_id])
            ready = next_ready
        if seen != len(steps):
            raise PlanValidationError("Circular dependency detected during scheduling.")
        return batches


def _lookup_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current
