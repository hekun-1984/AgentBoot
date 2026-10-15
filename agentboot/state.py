"""LangGraph 状态。共享字段用合并，轨迹用追加，避免并行子图互相覆盖。"""

import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def merge_dict(left: dict | None, right: dict | None) -> dict:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    shared_context: dict
    output: dict
    steps: int
    error: str | None
    error_kind: str | None
    failed: bool
    trace: list
    started_at: float
    state_before: dict


class SubgraphState(TypedDict, total=False):
    user_input: str
    shared_context: dict
    output: dict
    trace: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]


class GraphState(TypedDict, total=False):
    user_input: str
    target: str
    shared_context: Annotated[dict, merge_dict]
    plan: dict
    wave_index: int
    step_count: int
    results: Annotated[dict, merge_dict]
    trace: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    final_output: dict
    route_reason: str
    force_guard: bool
