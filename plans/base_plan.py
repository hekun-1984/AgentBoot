"""计划模板基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentboot.models import ExecutionPlan


class BasePlan(ABC):
    """可复用的执行计划模板。agent 必须等于路由表中的 Agent 名称。"""

    agent: str = ""

    @abstractmethod
    def build(self, context: dict[str, Any]) -> ExecutionPlan:
        """根据本次输入生成计划。"""
