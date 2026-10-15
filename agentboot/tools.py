"""Tool 注册。Tool 只做原子读写，不决定业务流程。"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import StructuredTool


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
