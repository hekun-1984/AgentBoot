"""Agent 契约。输入、输出、工具集、最大步数和失败策略都落在这里。"""

from __future__ import annotations

from dataclasses import dataclass, field


FAILURE_POLICIES = ("abort", "return_partial", "fallback", "escalate")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type_name: str = "str"
    description: str = ""
    required: bool = False


@dataclass
class SkillSpec:
    """一个 Agent 的可测试契约，来自 SKILL.md 或类属性。"""

    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    max_steps: int = 6
    on_failure: str = "return_partial"
    inputs: dict[str, FieldSpec] = field(default_factory=dict)
    outputs: dict[str, FieldSpec] = field(default_factory=dict)
    system_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Agent 名称不能为空")
        if self.max_steps < 1:
            raise ValueError(f"{self.name} 的 max_steps 必须大于 0")
        if self.on_failure not in FAILURE_POLICIES:
            allowed = ", ".join(FAILURE_POLICIES)
            raise ValueError(f"{self.name} 的 on_failure 必须是 {allowed}")


@dataclass
class AgentResult:
    output: dict
    trace: list
    error: str | None
    error_kind: str | None
    shared_context: dict
    steps: int
