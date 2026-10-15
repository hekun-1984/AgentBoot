"""子图抽象基类。SkillGraph：固定流程的业务单元，类似 Service。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from agentboot.exceptions import ExecutionError
from agentboot.models import Step

CapabilityRunner = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class BaseSubGraph(ABC):
    """子图基类。通过 class attributes 声明依赖关系、并行组和可调用能力。"""

    name: str = "base_subgraph"
    version: str = "1.0.0"
    description: str = ""
    depends_on: list[str] = []
    parallel_group: str | None = None
    allowed_skills: list[str] | None = None

    def bind_runner(self, runner: CapabilityRunner) -> None:
        """由调度器注入能力调用器，子图通过 call_skill 使用 Capability。"""
        self._runner = runner

    async def call_skill(self, name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """调用一个已注册能力。会同时受子图声明和路由白名单约束。"""
        runner = getattr(self, "_runner", None)
        if runner is None:
            raise ExecutionError(f"SubGraph '{self.name}' has no capability runner.")
        result = await runner(name, payload or {})
        if not isinstance(result, dict):
            raise ExecutionError(f"Skill '{name}' must return a dict.")
        return result

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """执行子图业务逻辑。context 包含原始输入和已完成步骤的结果。"""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name or self.name == "base_subgraph":
            errors.append("SubGraph name cannot be empty.")
        return errors

    def to_step(self) -> Step:
        return Step(
            id=self.name,
            type="subgraph",
            depends_on=list(self.depends_on),
            parallel_group=self.parallel_group,
            config={"allowed_skills": list(self.allowed_skills)} if self.allowed_skills else {},
        )
