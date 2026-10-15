"""Agent 抽象基类。OrchestratorGraph：定义 run / plan / validate。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentboot.exceptions import PlanValidationError, RegistrationError
from agentboot.graph import BaseSubGraph
from agentboot.models import ExecutionPlan, ExecutionResult


class BaseAgent(ABC):
    """业务 Agent。在构造函数里注册自己的子图和能力。"""

    name: str = ""
    description: str = ""

    def __init__(self) -> None:
        if not self.name:
            raise RegistrationError(f"{type(self).__name__} is missing name.")
        self._subgraphs: dict[str, type[BaseSubGraph]] = {}
        self._skills: dict[str, type] = {}

    def register_subgraph(self, graph_class: type[BaseSubGraph]) -> None:
        graph = graph_class()
        errors = graph.validate()
        if errors:
            raise RegistrationError("; ".join(errors))
        if graph.name in self._subgraphs:
            raise RegistrationError(
                f"SubGraph '{graph.name}' already registered on '{self.name}'."
            )
        self._subgraphs[graph.name] = graph_class

    def register_skill(self, skill_class: type) -> None:
        skill_name = getattr(skill_class, "name", "") or skill_class.__name__
        existing = self._skills.get(skill_name)
        if existing is not None and existing is not skill_class:
            raise RegistrationError(f"Skill '{skill_name}' already registered on '{self.name}'.")
        self._skills[skill_name] = skill_class

    @property
    def subgraphs(self) -> dict[str, type[BaseSubGraph]]:
        return dict(self._subgraphs)

    @property
    def skills(self) -> dict[str, type]:
        return dict(self._skills)

    def sample_context(self) -> dict[str, Any] | None:
        """提供示例输入时，python main.py 会在启动后跑一次。"""
        return None

    @abstractmethod
    async def execute_plan(self, context: dict[str, Any]) -> ExecutionPlan:
        """生成执行计划。模型规划失败时也会回到这里。"""

    async def plan(self, context: dict[str, Any], planner: Any | None = None) -> ExecutionPlan:
        if planner is not None:
            return await planner.plan(self, context)
        return await self.execute_plan(context)

    def validate(self, plan: ExecutionPlan) -> bool:
        """确认计划里的子图和能力都属于当前 Agent。"""
        for step in plan.steps:
            if step.type == "subgraph" and step.id not in self._subgraphs:
                raise PlanValidationError(
                    f"Step '{step.id}' is not registered on agent '{self.name}'."
                )
            if step.type == "skill" and step.id not in self._skills:
                raise PlanValidationError(
                    f"Step '{step.id}' is not registered on agent '{self.name}'."
                )
        return True

    async def run(self, context: dict[str, Any], app: Any) -> ExecutionResult:
        """通过应用容器执行当前 Agent。"""
        return await app.execute_plan(self.name, context)
