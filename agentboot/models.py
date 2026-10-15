"""数据模型定义。包含 ExecutionPlan、Step、ExecutionResult 等核心数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    """执行计划中的一个步骤节点。"""

    id: str
    type: str  # "subgraph" | "skill" | "llm" | "condition"
    depends_on: list[str] = field(default_factory=list)
    parallel_group: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    timeout: int | float | None = None

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class ExecutionPlan:
    """LLM 或计划模板生成的执行计划。"""

    plan_id: str
    agent: str
    steps: list[Step] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_parallel_groups(self) -> dict[str, list[Step]]:
        groups: dict[str, list[Step]] = {}
        for step in self.steps:
            key = step.parallel_group or "default"
            groups.setdefault(key, []).append(step)
        return groups


@dataclass
class ExecutionResult:
    """单次执行的结果。"""

    plan_id: str
    status: str  # "success" | "failed"
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
