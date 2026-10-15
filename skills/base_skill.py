"""能力抽象基类。Capability：原子能力，类似 DAO。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """可复用能力。子类用 name 作为白名单和注册 ID。"""

    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """执行原子能力。返回字典结果。"""

    def validate(self) -> list[str]:
        if not self.name:
            return ["Skill name cannot be empty."]
        return []
