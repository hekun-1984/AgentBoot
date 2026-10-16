"""Tool 注册。Tool 只做原子读写，不决定业务流程。"""

from __future__ import annotations

import inspect
import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from agentboot.exceptions import ToolReplayError

_REPLAY: ContextVar[dict[str, list] | None] = ContextVar("agentboot_tool_replay", default=None)
_EXECUTE = object()


def tool_outputs(record: dict) -> dict[str, list]:
    """按工具名收集当时的返回值，顺序与记录中的调用顺序一致。"""
    queues: dict[str, list] = {}

    def walk(spans: list) -> None:
        for span in spans or []:
            if isinstance(span, dict) and span.get("kind") == "tool" and span.get("name"):
                queues.setdefault(str(span["name"]), []).append(span.get("output"))
            if isinstance(span, dict):
                walk(span.get("children") or [])

    trace = record.get("trace")
    if not trace and isinstance(record.get("output"), dict):
        trace = record["output"].get("trace")
    walk(list(trace or []))
    return queues


class tool_replay:
    """重放期间，工具返回记录里的结果，不再调用原函数。"""

    def __init__(self, record: dict) -> None:
        self._queues = tool_outputs(record)
        self._token: Any = None

    def __enter__(self) -> "tool_replay":
        self._token = _REPLAY.set(self._queues)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _REPLAY.reset(self._token)


def frozen_or_execute(name: str) -> Any:
    queues = _REPLAY.get()
    if queues is None:
        return _EXECUTE
    pending = queues.get(name) or []
    if not pending:
        raise ToolReplayError(f"重放记录里没有工具 {name} 的下一次调用")
    return pending.pop(0)


def _replay_payload(frozen: Any) -> str:
    if isinstance(frozen, str):
        return frozen
    return json.dumps({"ok": True, "data": frozen}, ensure_ascii=False)


def _replay_data(frozen: Any) -> Any:
    if isinstance(frozen, str):
        try:
            payload = json.loads(frozen)
        except json.JSONDecodeError:
            return {"message": frozen}
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload
    return frozen


@dataclass
class ToolEntry:
    name: str
    description: str
    fn: Callable[..., Any]
    lc_tool: StructuredTool


def tool(name: str, description: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """把函数登记为原子能力。LangGraph 的 ToolNode 调用的是对应 StructuredTool。"""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        def runner(**kwargs: Any) -> str:
            frozen = frozen_or_execute(name)
            if frozen is not _EXECUTE:
                return _replay_payload(frozen)
            result = fn(**kwargs)
            return json.dumps({"ok": True, "data": result}, ensure_ascii=False)

        runner.__name__ = name
        runner.__doc__ = description
        runner.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        runner.__annotations__ = dict(getattr(fn, "__annotations__", {}))
        entry = ToolEntry(
            name=name,
            description=description,
            fn=fn,
            lc_tool=StructuredTool.from_function(runner, name=name, description=description),
        )
        fn.__agentboot_tool__ = entry  # type: ignore[attr-defined]
        return fn

    return decorate
