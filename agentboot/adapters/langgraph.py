"""可选的 LangGraph 执行器。

默认调度器不依赖 LangGraph。AGENTBOOT_ENGINE=langgraph 时，
同一份 ExecutionPlan 会编译成 StateGraph：每个 Step 是一个节点，depends_on 变成边。
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from agentboot.exceptions import ConfigurationError
from agentboot.models import ExecutionPlan, ExecutionResult, Step, StepStatus
from agentboot.scheduler import Scheduler, hydrate_plan


def ensure_langgraph() -> tuple[Any, Any, Any]:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ConfigurationError(
            'LangGraph engine requires an optional dependency. Install with: pip install -e ".[langgraph]"'
        ) from exc
    return START, END, StateGraph


def _merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class GraphState(TypedDict):
    payload: Annotated[dict[str, Any], _merge_dict]
    errors: Annotated[dict[str, str], _merge_dict]
    trace: Annotated[list[dict[str, Any]], operator.add]


class LangGraphExecutor:
    """把 ExecutionPlan 编译为 LangGraph StateGraph 后执行。"""

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    async def execute(
        self,
        plan: ExecutionPlan,
        context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        start, end, state_graph = ensure_langgraph()
        started = time.monotonic()
        context = dict(context or {})
        hydrate_plan(plan, self.scheduler.registry)
        builder = state_graph(GraphState)
        for step in plan.steps:
            builder.add_node(step.id, self._node(step, plan.agent))
        depended: set[str] = set()
        for step in plan.steps:
            if not step.depends_on:
                builder.add_edge(start, step.id)
            for dep_id in step.depends_on:
                builder.add_edge(dep_id, step.id)
                depended.add(dep_id)
        for step in plan.steps:
            if step.id not in depended:
                builder.add_edge(step.id, end)
        compiled = builder.compile()
        final = await compiled.ainvoke(
            {"payload": context, "errors": {}, "trace": []},
            config={"recursion_limit": max(50, len(plan.steps) * 4)},
        )
        step_ids = {step.id for step in plan.steps}
        errors = dict(final.get("errors") or {})
        results = {
            key: value
            for key, value in dict(final.get("payload") or {}).items()
            if key in step_ids and key not in errors
        }
        trace = [{"event": "engine", "engine": "langgraph"}, *list(final.get("trace") or [])]
        duration_ms = int((time.monotonic() - started) * 1000)
        return ExecutionResult(
            plan_id=plan.plan_id,
            status="failed" if errors else "success",
            results=results,
            errors=errors,
            trace=trace,
            duration_ms=duration_ms,
        )

    def _node(self, step: Step, agent_name: str):
        async def node(state: GraphState) -> dict[str, Any]:
            errors = state.get("errors") or {}
            if any(dep_id in errors for dep_id in step.depends_on):
                return {
                    "trace": [
                        {
                            "step": step.id,
                            "status": StepStatus.SKIPPED.value,
                            "reason": "upstream_failed",
                        }
                    ]
                }
            try:
                output = await self.scheduler.run_step(
                    step,
                    dict(state.get("payload") or {}),
                    agent_name,
                )
            except Exception as exc:
                return {
                    "errors": {step.id: str(exc)},
                    "trace": [
                        {
                            "step": step.id,
                            "status": StepStatus.FAILED.value,
                            "error": str(exc),
                        }
                    ],
                }
            duration_ms = output.pop("__duration_ms", None)
            return {
                "payload": {step.id: output},
                "trace": [
                    {
                        "step": step.id,
                        "status": StepStatus.SUCCESS.value,
                        "duration_ms": duration_ms,
                    }
                ],
            }

        return node
