"""模型适配。未配置模型时，Agent 走 SKILL 对应的 offline_step。"""

from __future__ import annotations

import os
from typing import Any, Callable, Protocol

from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.messages import AIMessage

_UNAVAILABLE = (ModelConnectionError, ModelTimeoutError, ModelRateLimitError, ModelAPIError)


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


def is_unavailable(exc: BaseException) -> bool:
    """模型连不上、超时、限流或服务端错误时可以换备选。答案不合格不算。"""
    if isinstance(exc, (OutputParserException, ModelInvalidRequestError)):
        return False
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _UNAVAILABLE):
            return True
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        status = getattr(current, "status_code", None)
        if not isinstance(status, int):
            response = getattr(current, "response", None)
            status = getattr(response, "status_code", None)
        if isinstance(status, int) and (status in {408, 429} or status >= 500):
            return True
        current = current.__cause__
    return False


def call_with_fallback(candidates: list[tuple[str, Any]], call: Callable[[Any], Any]) -> tuple[str, Any]:
    """按顺序调用。只有当前候选不可用时才换下一个。"""
    last: BaseException | None = None
    for index, (name, client) in enumerate(candidates):
        try:
            return name, call(client)
        except Exception as exc:
            last = exc
            if is_unavailable(exc) and index < len(candidates) - 1:
                continue
            raise
    if last is not None:
        raise last
    raise RuntimeError("没有可调用的模型")


def _stamp(message: AIMessage, option: str) -> AIMessage:
    metadata = dict(getattr(message, "response_metadata", None) or {})
    metadata["model_option"] = option
    message.response_metadata = metadata
    return message


class LangChainModel:
    """把 LangChain chat model 绑到 Agent 的工具白名单上。

    结束回答时走 with_structured_output，schema 来自 SKILL 的 output 字段。
    候选按默认模型在前的顺序排列，不可用时换下一个。
    """

    def __init__(self, chat_model: Any = None, *, candidates: list[tuple[str, Any]] | None = None) -> None:
        if candidates is not None:
            self.candidates = [(name, model) for name, model in candidates]
        elif chat_model is not None:
            self.candidates = [("default", chat_model)]
        else:
            raise ValueError("缺少模型")
        if not self.candidates:
            raise ValueError("缺少模型")
        self.chat_model = self.candidates[0][1]
        self.last_option = self.candidates[0][0]

    def invoke(self, messages: list, tools: list) -> AIMessage:
        def call(model: Any) -> AIMessage:
            bound = model.bind_tools(tools) if tools else model
            message = bound.invoke(messages)
            if isinstance(message, AIMessage):
                return message
            return AIMessage(content=getattr(message, "content", str(message)))

        name, message = call_with_fallback(self.candidates, call)
        self.last_option = name
        self.chat_model = dict(self.candidates)[name]
        return _stamp(message, name)

    def invoke_structured(self, messages: list, schema: Any) -> dict:
        def call(model: Any) -> Any:
            return model.with_structured_output(schema).invoke(messages)

        name, result = call_with_fallback(self.candidates, call)
        self.last_option = name
        self.chat_model = dict(self.candidates)[name]
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
