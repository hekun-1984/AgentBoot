"""Agent 离线循环共用的小函数。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from agentboot.agent import tool_payloads


def payloads(messages: list) -> dict[str, dict]:
    return tool_payloads(messages)


def data_of(found: dict[str, dict], name: str) -> dict:
    payload = found.get(name) or {}
    if not payload.get("ok"):
        return {}
    data = payload.get("data") or {}
    return data if isinstance(data, dict) else {"value": data}


def calls(*items: tuple[str, dict, str]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            for name, args, call_id in items
        ],
    )


def final(body: dict) -> AIMessage:
    return AIMessage(content=json.dumps(body, ensure_ascii=False))
