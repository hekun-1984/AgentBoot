"""主图：只决定用哪些子图、按什么波次执行，并保留兜底。"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import Field, create_model

from agentboot.agent import parse_json_object
from agentboot.state import GraphState
from agentboot.tracing import new_span, snapshot, start_timer


def route_plan_model(names: list[str]):
    """路由计划的结构。subgraph 只能取这张主图已经注册的名字。"""
    allowed = tuple(dict.fromkeys(name for name in names if name))
    if not allowed:
        allowed = ("",)
    step = create_model(
        "RouteStep",
        subgraph=(Literal[allowed], Field(description="已注册的子图")),
        parallel_group=(str | None, Field(default=None, description="同一组则并行")),
    )
    return create_model(
        "RoutePlan",
        reason=(str, Field(description="选择原因")),
        steps=(list[step], Field(description="执行步骤")),
    )


def group_waves(steps: list[dict]) -> list[list[str]]:
    waves: list[list[str]] = []
    positions: dict[str, int] = {}
    for step in steps:
        name = step["subgraph"]
        group = step.get("parallel_group")
        if not group:
            waves.append([name])
            continue
        if group not in positions:
            positions[group] = len(waves)
            waves.append([])
        if name not in waves[positions[group]]:
            waves[positions[group]].append(name)
    return waves


class BaseGraph:
    """主图白名单来自 subgraphs。安全筛查和空计划兜底不交给模型。"""

    name = ""
    description = ""
    subgraphs: list[str] = []
    fallback_subgraph = ""
    emergency_subgraph = ""
    max_steps = 6
    emergency_symptoms: tuple[str, ...] = ()
    guard_field = ""
    guard_values: tuple = ()
    guard_subgraph = ""
    pause_for_human = False

    def __init__(self) -> None:
        self.model = None
        self._tools: dict[str, Any] = {}
        self._compiled_subgraphs: dict[str, Any] = {}
        self.catalog: list[dict[str, str]] = []
        self.graph = None
        self.checkpointer = None

    def bind(
        self,
        subgraphs: dict[str, Any],
        tools: dict[str, Any],
        model: Any = None,
        *,
        checkpointer: Any = None,
    ) -> "BaseGraph":
        missing = [name for name in self.subgraphs if name not in subgraphs]
        if missing:
            joined = ", ".join(missing)
            raise KeyError(f"主图 {self.name} 缺少子图: {joined}")
        self._compiled_subgraphs = {name: subgraphs[name] for name in self.subgraphs}
        self._tools = tools
        self.model = model
        self.catalog = [
            {"name": name, "description": subgraphs[name].description}
            for name in self.subgraphs
        ]
        if checkpointer is None and self.pause_for_human:
            checkpointer = InMemorySaver()
        self.checkpointer = checkpointer
        self.graph = self.compile()
        return self

    def offline_route(self, user_input: str, catalog: list[dict], shared_context: dict) -> dict:
        return {"steps": [], "reason": "未配置路由模型"}

    def validate_plan(self, raw: dict | None) -> dict:
        raw = raw or {}
        allowed = set(self.subgraphs)
        cleaned = []
        for step in raw.get("steps") or []:
            if not isinstance(step, dict):
                continue
            name = step.get("subgraph")
            if name not in allowed:
                continue
            cleaned.append({"subgraph": name, "parallel_group": step.get("parallel_group")})
        reason = str(raw.get("reason") or "")
        if len(cleaned) > self.max_steps:
            cleaned = cleaned[: self.max_steps]
            reason = f"{reason} | 超过最大步数，已截断".strip(" |")
        if not cleaned and self.fallback_subgraph in allowed:
            cleaned = [{"subgraph": self.fallback_subgraph, "parallel_group": None}]
            reason = f"{reason} | 空计划兜底:{self.fallback_subgraph}".strip(" |")
        return {"waves": group_waves(cleaned), "steps": cleaned, "reason": reason}

    def build_plan(self, state: GraphState) -> dict:
        context = state.get("shared_context") or {}
        if context.get("force_emergency") and self.emergency_subgraph:
            return self.validate_plan(
                {
                    "steps": [{"subgraph": self.emergency_subgraph}],
                    "reason": "危急值或危急症状命中，只进入紧急升级",
                }
            )
        raw: dict | None
        if self.model is not None:
            try:
                raw = self._llm_route(str(state.get("user_input") or ""), context)
            except Exception as exc:
                raw = {"steps": [], "reason": f"路由模型失败: {exc}"}
        else:
            raw = self.offline_route(str(state.get("user_input") or ""), self.catalog, dict(context))
        return self.validate_plan(raw)

    def render_answer(self, state: GraphState) -> str:
        results = state.get("results") or {}
        if not results:
            return "没有可执行的子图。"
        return json.dumps(results, ensure_ascii=False)

    def screen(self, user_input: str, shared_context: dict) -> dict:
        """路由前的固定检查。子类在这里写入领域字段，需要时把 force_emergency 设为 True。"""
        force = any(word in user_input for word in self.emergency_symptoms)
        return {"user_input": user_input, "force_emergency": force}

    def compile(self):
        def safety(state: GraphState) -> dict:
            started = start_timer()
            text = str(state.get("user_input") or "")
            before = snapshot(state.get("shared_context") or {})
            context = self.screen(text, dict(state.get("shared_context") or {}))
            context.setdefault("user_input", text)
            span = new_span(
                kind="graph",
                name="safety_screen",
                span_input=text,
                span_output={key: context.get(key) for key in ("force_emergency", "intake")},
                started=started,
                state_before=before,
                state_after={**before, **context},
            )
            return {"shared_context": context, "trace": [span]}

        def route(state: GraphState) -> dict:
            started = start_timer()
            plan = self.build_plan(state)
            span = new_span(
                kind="graph",
                name="route",
                span_input={"user_input": state.get("user_input"), "catalog": [item["name"] for item in self.catalog]},
                span_output=plan,
                started=started,
                state_before=state.get("shared_context"),
                state_after=state.get("shared_context"),
            )
            return {
                "plan": plan,
                "route_reason": plan["reason"],
                "wave_index": 0,
                "step_count": 0,
                "force_guard": False,
                "trace": [span],
            }

        def fanout(state: GraphState) -> str | list[Send]:
            plan = state.get("plan") or {}
            waves = plan.get("waves") or []
            index = int(state.get("wave_index") or 0)
            if index >= len(waves):
                return "guard"
            if int(state.get("step_count") or 0) >= self.max_steps and not state.get("force_guard"):
                return "guard"
            context = dict(state.get("shared_context") or {})
            return [
                Send(
                    name,
                    {
                        "user_input": state.get("user_input"),
                        "shared_context": context,
                    },
                )
                for name in waves[index]
            ]

        def join(state: GraphState) -> dict:
            return {
                "wave_index": int(state.get("wave_index") or 0) + 1,
                "step_count": int(state.get("step_count") or 0) + 1,
                "force_guard": False,
            }

        def guard(state: GraphState) -> dict:
            context = state.get("shared_context") or {}
            target = self.guard_subgraph
            ran = set(state.get("results") or {})
            value = context.get(self.guard_field) if self.guard_field else None
            if target and value in self.guard_values and target not in ran:
                plan = self.validate_plan(
                    {
                        "steps": [{"subgraph": target}],
                        "reason": f"{state.get('route_reason') or ''} | 字段兜底:{self.guard_field}={value}".strip(" |"),
                    }
                )
                return {"plan": plan, "wave_index": 0, "force_guard": True, "route_reason": plan["reason"]}
            return {"force_guard": False}

        def after_guard(state: GraphState) -> str:
            if state.get("force_guard"):
                decided = "human_decision" in (state.get("shared_context") or {})
                if self.pause_for_human and not decided:
                    return "wait"
                return "fanout"
            return "aggregate"

        def wait(state: GraphState) -> dict:
            context = state.get("shared_context") or {}
            value = context.get(self.guard_field)
            decision = interrupt(
                {
                    "status": "waiting_human",
                    "field": self.guard_field,
                    "value": value,
                    "subgraph": self.guard_subgraph,
                    "user_input": state.get("user_input") or "",
                    "message": f"等待人工决定是否继续 {self.guard_subgraph}",
                }
            )
            return {"shared_context": {"human_decision": decision}}

        def aggregate(state: GraphState) -> dict:
            started = start_timer()
            answer = self.render_answer(state)
            final = {
                "answer": answer,
                "shared_context": dict(state.get("shared_context") or {}),
                "results": dict(state.get("results") or {}),
                "plan": dict(state.get("plan") or {}),
                "route_reason": state.get("route_reason") or "",
                "trace": list(state.get("trace") or []),
                "errors": list(state.get("errors") or []),
            }
            span = new_span(
                kind="graph",
                name=self.name,
                span_input=state.get("user_input"),
                span_output={"answer": answer, "plan": final["plan"]},
                started=started,
                state_before=None,
                state_after=final["shared_context"],
            )
            final["trace"] = list(final["trace"]) + [span]
            return {"final_output": final, "trace": [span]}

        builder = StateGraph(GraphState)
        builder.add_node("safety", safety)
        builder.add_node("route", route)
        builder.add_node("fanout", lambda state: {})
        builder.add_node("join", join)
        builder.add_node("guard", guard)
        builder.add_node("aggregate", aggregate)
        for name, compiled in self._compiled_subgraphs.items():
            builder.add_node(name, compiled.graph)
            builder.add_edge(name, "join")
        builder.add_edge(START, "safety")
        builder.add_edge("safety", "route")
        builder.add_edge("route", "fanout")
        builder.add_conditional_edges("fanout", fanout, [*self.subgraphs, "guard"])
        builder.add_edge("join", "fanout")
        if self.pause_for_human:
            builder.add_node("wait", wait)
            builder.add_conditional_edges("guard", after_guard, ["fanout", "aggregate", "wait"])
            builder.add_edge("wait", "fanout")
        else:
            builder.add_conditional_edges("guard", after_guard, ["fanout", "aggregate"])
        builder.add_edge("aggregate", END)
        if self.checkpointer is None:
            return builder.compile()
        return builder.compile(checkpointer=self.checkpointer)

    def _llm_route(self, user_input: str, shared_context: dict) -> dict:
        catalog = json.dumps(self.catalog, ensure_ascii=False)
        system = (
            "你是主图路由器。只能从给定子图白名单里选择，不能发明子图，也不能规定子图内部步骤。"
            "同一 parallel_group 的子图会并行。只返回 JSON："
            '{"reason":"...","steps":[{"subgraph":"名字","parallel_group":"可选"}]}'
        )
        human = json.dumps(
            {"user_input": user_input, "catalog": json.loads(catalog), "shared_context": shared_context},
            ensure_ascii=False,
        )
        message_list = [
            SystemMessage(content=system),
            HumanMessage(content=human),
        ]
        structured = getattr(self.model, "invoke_structured", None)
        if callable(structured):
            data = structured(message_list, route_plan_model(list(self.subgraphs)))
            if hasattr(data, "model_dump"):
                data = data.model_dump()
            if not isinstance(data, dict):
                raise TypeError("路由结果不是对象")
            return data
        message = self.model.invoke(message_list, [])
        return parse_json_object(message.content if isinstance(message.content, str) else str(message.content))
