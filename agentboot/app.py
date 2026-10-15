"""AgentBoot 应用核心。负责初始化注册中心、加载配置、启动调度器。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from agentboot.config import Settings
from agentboot.exceptions import PlanValidationError, RegistrationError
from agentboot.logger import get_logger, setup_logging
from agentboot.models import ExecutionResult
from agentboot.planner import Planner, discover_plans
from agentboot.registry import Registry
from agentboot.router import Router
from agentboot.scheduler import Scheduler, hydrate_plan
from agentboot.validator import PlanValidator

logger = get_logger(__name__)


class AgentApp:
    """Agent 应用主类。类似 Spring Boot 的 ApplicationContext。"""

    def __init__(self, app_name: str | None = None, project_root: Path | str | None = None) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        root = str(self.project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        self.settings = Settings(project_root=self.project_root)
        setup_logging(self.settings.log_level)
        self.app_name = app_name or self.settings.app_name
        self.registry = Registry()
        self.router = Router(settings=self.settings)
        self.scheduler = Scheduler(
            registry=self.registry,
            router=self.router,
            llm_model=self.settings.llm_model,
        )
        self.planner = Planner(settings=self.settings, registry=self.registry)
        self.scheduler.llm_client = self.planner.llm_client
        self.executor: Any = self._build_executor()
        self._ready = False

    def _build_executor(self) -> Any:
        if self.settings.engine != "langgraph":
            return self.scheduler
        from agentboot.adapters.langgraph import LangGraphExecutor, ensure_langgraph

        ensure_langgraph()
        return LangGraphExecutor(self.scheduler)

    def register_agent(self, agent_class: type) -> None:
        """注册一个 Agent 类。类似 Spring 的组件注册。"""
        logger.info("Registering agent: %s", agent_class.__name__)
        self.registry.register(agent_class)

    def register_skill(self, skill_class: type) -> None:
        """注册一个可复用能力。"""
        logger.info("Registering skill: %s", getattr(skill_class, "name", skill_class.__name__))
        self.registry.register_skill(skill_class)

    def bootstrap(self) -> None:
        """加载白名单和计划模板，不阻塞等待请求。"""
        self.router.load_whitelist()
        self.planner.templates = discover_plans()
        self._warn_missing_skills()
        self._ready = True

    def run(self, demo: bool | None = None) -> None:
        """启动应用。demo 默认跟随 AGENTBOOT_DEMO。"""
        logger.info("Starting %s...", self.app_name)
        self.bootstrap()
        use_demo = self.settings.demo if demo is None else demo
        exit_code = asyncio.run(self._serve(use_demo))
        if exit_code:
            raise SystemExit(exit_code)

    async def _serve(self, demo: bool) -> int:
        await self.scheduler.start()
        logger.info("%s started successfully.", self.app_name)
        if not demo:
            return 0
        return await self._run_samples()

    def execute(self, agent_name: str, context: dict[str, Any] | None = None) -> ExecutionResult:
        """同步提交一次执行。不要在已经运行的事件循环里调用。"""
        self.bootstrap()
        return asyncio.run(self.execute_plan(agent_name, context))

    async def execute_plan(
        self,
        plan_id: str,
        context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """外部调用入口：按 Agent 名称生成计划并运行。"""
        if not self._ready:
            self.bootstrap()
        context = dict(context or {})
        agent = self.registry.get_agent_instance(plan_id)
        if agent is None:
            raise RegistrationError(f"Agent '{plan_id}' is not registered.")
        plan = await self.planner.plan(agent, context)
        hydrate_plan(plan, self.registry)
        agent.validate(plan)
        self._check_route_limits(plan)
        validator = PlanValidator(
            set(self.registry.list_subgraphs()),
            set(self.registry.list_skills()),
        )
        validator.validate(plan)
        return await self.executor.execute(plan, context)

    def _check_route_limits(self, plan: Any) -> None:
        max_steps = self.router.get_route(plan.agent).get("max_steps")
        if max_steps is not None and len(plan.steps) > int(max_steps):
            raise PlanValidationError(
                f"Plan '{plan.plan_id}' has {len(plan.steps)} steps, limit is {max_steps}."
            )

    def _warn_missing_skills(self) -> None:
        for name in self.registry.list_subgraphs():
            graph_class = self.registry.get_subgraph(name)
            allowed = list(getattr(graph_class, "allowed_skills", None) or [])
            for skill_name in allowed:
                if self.registry.get_skill(skill_name) is None:
                    logger.warning("Subgraph %s allows missing skill %s.", name, skill_name)

    async def _run_samples(self) -> int:
        failed = 0
        ran = 0
        for name in self.registry.list_agents():
            agent = self.registry.get_agent_instance(name)
            sample = agent.sample_context()
            if sample is None:
                continue
            ran += 1
            result = await self.execute_plan(name, sample)
            _print_result(agent, result)
            if result.status != "success":
                failed += 1
        if ran == 0:
            logger.info("No sample context found. Application is idle.")
        return 1 if failed else 0


def _print_result(agent: Any, result: ExecutionResult) -> None:
    batches: list[list[str]] = []
    for item in result.trace:
        if "batches" in item:
            batches = item["batches"]
    print(f"\n=== {agent.name} / {result.plan_id} ===")
    print(f"status: {result.status}  duration_ms: {result.duration_ms}")
    if batches:
        print("batches: " + " -> ".join(", ".join(batch) for batch in batches))
    if result.errors:
        print("errors: " + json.dumps(result.errors, ensure_ascii=False))
    print(json.dumps(result.results, ensure_ascii=False, indent=2, default=str))
