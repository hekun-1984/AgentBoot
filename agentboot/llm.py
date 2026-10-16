"""模型适配。未配置模型时，Agent 走 SKILL 对应的 offline_step。"""

from __future__ import annotations

import os
from typing import Any, Protocol

from langchain_core.messages import AIMessage


class ChatModel(Protocol):
    def invoke(self, messages: list, tools: list) -> AIMessage:
        """返回带 content 或 tool_calls 的 AIMessage。"""


class ScriptedModel:
    """按脚本回放模型输出，供 Agent 单测使用。"""

    def __init__(self, replies: list[AIMessage]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, messages: list, tools: list) -> AIMessage:
        self.calls += 1
        if not self._replies:
            return AIMessage(content="{}")
        return self._replies.pop(0)


class LangChainModel:
    """把 LangChain chat model 绑到 Agent 的工具白名单上。

    结束回答时走 with_structured_output，schema 来自 SKILL 的 output 字段。
    """

    def __init__(self, chat_model: Any) -> None:
        self.chat_model = chat_model

    def invoke(self, messages: list, tools: list) -> AIMessage:
        model = self.chat_model.bind_tools(tools) if tools else self.chat_model
        message = model.invoke(messages)
        if isinstance(message, AIMessage):
            return message
        return AIMessage(content=getattr(message, "content", str(message)))

    def invoke_structured(self, messages: list, schema: Any) -> dict:
        result = self.chat_model.with_structured_output(schema).invoke(messages)
        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif isinstance(result, dict):
            data = result
        else:
            raise TypeError("结构化输出不是对象")
        if not isinstance(data, dict):
            raise TypeError("结构化输出不是对象")
        return data


def load_model(model_name: str | None = None) -> ChatModel | None:
    name = model_name if model_name is not None else os.getenv("AGENTBOOT_MODEL")
    if not name:
        return None
    from langchain.chat_models import init_chat_model

    return LangChainModel(init_chat_model(name))
