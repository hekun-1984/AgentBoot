"""Agent：一次模型循环加白名单 Tool。每个 Agent 自己就是一张 LangGraph。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agentboot.exceptions import AgentExecutionError, RegistrationError, SkillError
from agentboot.models import AgentResult, FieldSpec, SkillSpec
from agentboot.skill import load_skill
from agentboot.state import AgentState
from agentboot.tracing import message_brief, new_span, snapshot, start_timer, sum_tokens, usage_of


def parse_json_object(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出里没有 JSON 对象")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return data


def tool_payloads(messages: list) -> dict[str, dict]:
    """读取本轮 ToolMessage。成功时返回 {tool_name: payload}。"""
    found: dict[str, dict] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {"ok": False, "message": content}
        if not isinstance(payload, dict):
            payload = {"ok": True, "data": payload}
        found[message.name or ""] = payload
    return found


class BaseAgent:
    """子类用类属性声明名字，并用同目录 SKILL.md 声明契约。

    模型可用时走工具调用循环。未配置模型时调用 offline_step，循环结构相同。
    """

    name = ""
    description = ""
    tools: list[str] = []
    max_steps = 6
    on_failure = "return_partial"
    skill_path: str | None = None
    input_fields: dict[str, str] = {}
    output_fields: dict[str, str] = {}

    def __init__(self) -> None:
        self.spec = self._load_spec()
        if self.spec.name != self.name:
            raise SkillError(f"{self.__class__.__name__} 的 name 与 SKILL.md 不一致: {self.spec.name}")
        self.model = None
        self.tool_entries: dict[str, Any] = {}
        self.graph = None

    def _load_spec(self) -> SkillSpec:
        if self.skill_path:
            return load_skill(self.skill_path)
        inputs = {
            key: FieldSpec(name=key, description=text, required=False)
            for key, text in self.input_fields.items()
        }
        outputs = {
            key: FieldSpec(name=key, description=text, required=False)
            for key, text in self.output_fields.items()
        }
        return SkillSpec(
            name=self.name,
            description=self.description,
            tools=list(self.tools),
            max_steps=self.max_steps,
            on_failure=self.on_failure,
            inputs=inputs,
            outputs=outputs,
            system_prompt=self.description,
        )

    def bind(self, tool_entries: dict[str, Any], model: Any = None) -> "BaseAgent":
        missing = [name for name in self.spec.tools if name not in tool_entries]
        if missing:
            joined = ", ".join(missing)
            raise RegistrationError(f"Agent {self.name} 缺少 Tool: {joined}")
        self.tool_entries = {name: tool_entries[name] for name in self.spec.tools}
        self.model = model
        self.graph = self.compile(model)
        return self

    def offline_step(self, messages: list, shared_context: dict) -> AIMessage:
        raise AgentExecutionError(
            f"{self.name} 没有模型，也没有 offline_step",
            error_kind="llm",
            agent=self.name,
        )

    def fallback_output(self, shared_context: dict) -> dict:
        return {key: None for key in self.spec.outputs}

    def compile(self, model: Any = None):
        selected = self.model if model is None else model
        lc_tools = [entry.lc_tool for entry in self.tool_entries.values()]
        tool_node = ToolNode(lc_tools, handle_tool_errors=True) if lc_tools else None
        allowed = set(self.spec.tools)

        def prepare(state: AgentState) -> dict:
            context = dict(state.get("shared_context") or {})
            missing = [
                field.name
                for field in self.spec.inputs.values()
                if field.required and field.name != "user_input" and context.get(field.name) in (None, "")
            ]
            user_required = self.spec.inputs.get("user_input")
            user_text = _user_text(state.get("messages") or [])
            if user_required and user_required.required and not user_text:
                missing.append("user_input")
            before = snapshot(context)
            if missing:
                return self._fail(
                    state,
                    f"缺少输入: {', '.join(missing)}",
                    "state",
                    state_before=before,
                )
            return {"started_at": start_timer(), "state_before": before, "failed": False, "steps": 0}

        def model_node(state: AgentState) -> dict:
            if state.get("steps", 0) >= self.spec.max_steps:
                return self._fail(state, f"{self.name} 超过最大步数 {self.spec.max_steps}", "llm")
            started = start_timer()
            messages = list(state.get("messages") or [])
            try:
                if selected is not None:
                    ai_message = selected.invoke(messages, lc_tools)
                else:
                    ai_message = self.offline_step(messages, dict(state.get("shared_context") or {}))
            except AgentExecutionError:
                raise
            except Exception as exc:
                return self._fail(state, f"模型调用失败: {exc}", "llm")
            if not isinstance(ai_message, AIMessage):
                ai_message = AIMessage(content=str(ai_message))
            unknown = [call["name"] for call in ai_message.tool_calls if call["name"] not in allowed]
            span = new_span(
                kind="llm",
                name=self.name,
                span_input=[message_brief(item) for item in messages],
                span_output=message_brief(ai_message),
                started=started,
                tokens=usage_of(ai_message),
                state_before=state.get("state_before"),
                state_after=state.get("shared_context"),
                error=f"调用了未授权工具: {', '.join(unknown)}" if unknown else None,
                error_kind="llm" if unknown else None,
            )
            trace = list(state.get("trace") or [])
            trace.append(span)
            if unknown:
                return self._fail(
                    {**state, "trace": trace, "steps": state.get("steps", 0) + 1},
                    f"调用了未授权工具: {', '.join(unknown)}",
                    "llm",
                )
            return {"messages": [ai_message], "steps": state.get("steps", 0) + 1, "trace": trace}

        def route_after_model(state: AgentState) -> str:
            if state.get("failed"):
                return END
            last = (state.get("messages") or [None])[-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                if tool_node is None:
                    return END
                return "tools"
            return "parse"

        def observe(state: AgentState) -> dict:
            started = start_timer()
            messages = list(state.get("messages") or [])
            ai_message = _last_ai(messages)
            ids = {call["id"] for call in (ai_message.tool_calls if ai_message else [])}
            tool_messages = [
                item
                for item in messages
                if isinstance(item, ToolMessage) and (not ids or item.tool_call_id in ids)
            ]
            spans = []
            failed_names = []
            for item in tool_messages:
                failed = getattr(item, "status", None) == "error"
                if failed:
                    failed_names.append(item.name or "tool")
                spans.append(
                    new_span(
                        kind="tool",
                        name=item.name or "tool",
                        span_input=_tool_args(ai_message, item.tool_call_id),
                        span_output=item.content,
                        started=started,
                        error=item.content if failed else None,
                        error_kind="tool" if failed else None,
                        state_before=state.get("shared_context"),
                        state_after=state.get("shared_context"),
                    )
                )
            trace = list(state.get("trace") or [])
            trace.extend(spans)
            if failed_names:
                return self._fail(
                    {**state, "trace": trace},
                    f"Tool 调用失败: {', '.join(failed_names)}",
                    "tool",
                )
            return {"trace": trace}

        def parse(state: AgentState) -> dict:
            last = (state.get("messages") or [None])[-1]
            content = last.content if isinstance(last, AIMessage) else ""
            if not isinstance(content, str):
                content = str(content)
            try:
                data = parse_json_object(content)
            except (ValueError, json.JSONDecodeError) as exc:
                return self._fail(state, f"结构化输出失败: {exc}", "llm")
            if self.spec.outputs:
                data = {key: data.get(key) for key in self.spec.outputs}
            context = dict(state.get("shared_context") or {})
            context.update({key: value for key, value in data.items() if value is not None})
            return self._close(state, output=data, shared_context=context)

        builder = StateGraph(AgentState)
        builder.add_node("prepare", prepare)
        builder.add_node("model", model_node)
        builder.add_node("parse", parse)
        builder.add_edge(START, "prepare")
        builder.add_conditional_edges("prepare", lambda state: END if state.get("failed") else "model")
        if tool_node is not None:
            builder.add_node("tools", tool_node)
            builder.add_node("observe", observe)
            builder.add_conditional_edges("model", route_after_model)
            builder.add_edge("tools", "observe")
            builder.add_conditional_edges("observe", lambda state: END if state.get("failed") else "model")
        else:
            builder.add_conditional_edges(
                "model",
                lambda state: END if state.get("failed") else "parse",
            )
        builder.add_edge("parse", END)
        return builder.compile()

    def invoke(
        self,
        shared_context: dict | None = None,
        *,
        user_input: str = "",
        model: Any = None,
    ) -> AgentResult:
        context = dict(shared_context or {})
        if user_input:
            context.setdefault("user_input", user_input)
        graph = self.graph if model is None and self.graph is not None else self.compile(model)
        state = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=self.spec.system_prompt or self.spec.description),
                    HumanMessage(content=user_input or str(context.get("user_input") or "")),
                ],
                "shared_context": context,
                "output": {},
                "steps": 0,
                "trace": [],
                "failed": False,
            }
        )
        return AgentResult(
            output=dict(state.get("output") or {}),
            trace=list(state.get("trace") or []),
            error=state.get("error"),
            error_kind=state.get("error_kind"),
            shared_context=dict(state.get("shared_context") or {}),
            steps=int(state.get("steps") or 0),
        )

    def _close(
        self,
        state: AgentState,
        *,
        output: dict,
        shared_context: dict | None = None,
        error: str | None = None,
        error_kind: str | None = None,
    ) -> dict:
        context = dict(shared_context or state.get("shared_context") or {})
        children = list(state.get("trace") or [])
        span = new_span(
            kind="agent",
            name=self.name,
            span_input={"shared_context": state.get("state_before"), "user_input": _user_text(state.get("messages") or [])},
            span_output=output,
            started=float(state.get("started_at") or start_timer()),
            error=error,
            error_kind=error_kind,
            tokens=sum_tokens(children),
            state_before=state.get("state_before"),
            state_after=context,
            children=children,
        )
        update = {
            "output": output,
            "shared_context": context,
            "trace": [span],
            "error": error,
            "error_kind": error_kind,
            "failed": error is not None,
        }
        if error and self.spec.on_failure == "abort":
            raise AgentExecutionError(error, error_kind=error_kind or "llm", agent=self.name)
        return update

    def _fail(self, state: AgentState, message: str, error_kind: str, state_before: dict | None = None) -> dict:
        context = dict(state.get("shared_context") or {})
        if state_before is not None:
            state = {**state, "state_before": state_before, "started_at": state.get("started_at") or start_timer()}
        if self.spec.on_failure == "fallback":
            output = self.fallback_output(context)
        elif self.spec.on_failure == "escalate":
            output = {"needs_human": True, "error": message}
        else:
            output = {"error": message}
            for key, value in context.items():
                if key in self.spec.outputs:
                    output[key] = value
        if self.spec.outputs:
            for key in self.spec.outputs:
                output.setdefault(key, None)
        return self._close(state, output=output, shared_context=context, error=message, error_kind=error_kind)


def _user_text(messages: list) -> str:
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


def _last_ai(messages: list) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _tool_args(message: AIMessage | None, tool_call_id: str | None) -> dict:
    if message is None:
        return {}
    for call in message.tool_calls:
        if call.get("id") == tool_call_id:
            return dict(call.get("args") or {})
    return {}
