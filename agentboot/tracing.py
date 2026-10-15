"""执行轨迹。Agent span 下面挂 LLM 与 Tool 子 span。"""

from __future__ import annotations

import copy
import time
from typing import Any


def snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return repr(value)


def start_timer() -> float:
    return time.perf_counter()


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def new_span(
    *,
    kind: str,
    name: str,
    span_input: Any,
    span_output: Any,
    started: float,
    error: str | None = None,
    error_kind: str | None = None,
    tokens: dict | None = None,
    state_before: Any = None,
    state_after: Any = None,
    children: list | None = None,
) -> dict:
    return {
        "kind": kind,
        "name": name,
        "input": snapshot(span_input),
        "output": snapshot(span_output),
        "duration_ms": elapsed_ms(started),
        "tokens": dict(tokens or {}),
        "error": error,
        "error_kind": error_kind,
        "state_before": snapshot(state_before),
        "state_after": snapshot(state_after),
        "children": list(children or []),
    }


def sum_tokens(spans: list[dict]) -> dict:
    prompt = 0
    completion = 0
    for span in spans:
        tokens = span.get("tokens") or {}
        prompt += int(tokens.get("input_tokens") or 0)
        completion += int(tokens.get("output_tokens") or 0)
        nested = sum_tokens(span.get("children") or [])
        prompt += int(nested.get("input_tokens") or 0)
        completion += int(nested.get("output_tokens") or 0)
    return {"input_tokens": prompt, "output_tokens": completion}


def message_brief(message: Any) -> dict:
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = str(content)
    brief = {"type": getattr(message, "type", type(message).__name__), "content": content}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        brief["tool_calls"] = [
            {"name": call.get("name"), "args": call.get("args"), "id": call.get("id")}
            for call in tool_calls
        ]
    status = getattr(message, "status", None)
    if status:
        brief["status"] = status
    name = getattr(message, "name", None)
    if name:
        brief["name"] = name
    return brief


def usage_of(message: Any) -> dict:
    metadata = getattr(message, "usage_metadata", None) or {}
    if not isinstance(metadata, dict):
        return {}
    return {
        "input_tokens": int(metadata.get("input_tokens") or 0),
        "output_tokens": int(metadata.get("output_tokens") or 0),
    }
