"""Subgraph：固定边的能力模块。模型不能改这里的边。"""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from agentboot.agent import BaseAgent
from agentboot.exceptions import RegistrationError
from agentboot.tool import _EXECUTE, _replay_data, frozen_or_execute
from agentboot.state import SubgraphState
from agentboot.tracing import new_span, start_timer


class RunnableSubgraph:
    """已编译的子图。主图只拿得到名字、说明、写出字段和 invoke。"""

    def __init__(self, owner: "BaseSubgraph", graph: Any) -> None:
        self.name = owner.name
        self.description = owner.description
        self.writes = list(owner.writes)
        self.reads = list(getattr(owner, "reads", []) or [])
        self.graph = graph

    def invoke(self, state: dict) -> dict:
        context = dict(state.get("shared_context") or {})
        missing = [key for key in self.reads if context.get(key) in (None, "")]
        if missing:
            joined = ", ".join(missing)
            return {
                "user_input": state.get("user_input") or "",
                "shared_context": context,
                "output": {},
                "trace": [],
                "errors": [
                    {
                        "source": self.name,
                        "message": f"缺少读取字段: {joined}",
                        "error_kind": "state",
                    }
                ],
            }
        return self.graph.invoke(state)


class SubgraphRuntime:
    """给子图 build 方法用的节点工厂。Agent 和 Tool 都从白名单里取。"""

    def __init__(self, agents: dict[str, BaseAgent], tools: dict[str, Any]) -> None:
        self.agents = agents
        self.tools = tools

    def add_agent(self, builder: StateGraph, node_name: str, agent_name: str | None = None) -> None:
        agent = self.agents[agent_name or node_name]

        def node(state: SubgraphState, agent: BaseAgent = agent) -> dict:
            result = agent.invoke(state.get("shared_context"), user_input=str(state.get("user_input") or ""))
            context = dict(state.get("shared_context") or {})
            published = {
                key: value
                for key, value in result.output.items()
                if value is not None and key in agent.spec.outputs
            }
            context.update(published)
            output = dict(state.get("output") or {})
            output.update(published)
            errors = []
            if result.error:
                errors.append({"source": agent.name, "message": result.error, "error_kind": result.error_kind})
            return {
                "shared_context": context,
                "output": output,
                "trace": result.trace,
                "errors": errors,
            }

        builder.add_node(node_name, node)

    def add_sequence(self, builder: StateGraph, names: list[str]) -> str:
        if not names:
            raise RegistrationError("子图至少要有一个步骤")
        for name in names:
            if name not in self.agents:
                raise RegistrationError(f"子图引用了未注册的 Agent: {name}")
            self.add_agent(builder, name)
        builder.add_edge(START, names[0])
        for left, right in zip(names, names[1:]):
            builder.add_edge(left, right)
        return names[-1]

    def add_tool(
        self,
        builder: StateGraph,
        node_name: str,
        tool_name: str,
        *,
        read: dict[str, str],
        write: dict[str, str],
        json_args: tuple[str, ...] = (),
    ) -> None:
        if tool_name not in self.tools:
            raise RegistrationError(f"子图引用了未注册的 Tool: {tool_name}")
        entry = self.tools[tool_name]

        def node(state: SubgraphState, entry=entry) -> dict:
            started = start_timer()
            context = dict(state.get("shared_context") or {})
            kwargs: dict[str, Any] = {}
            for arg, key in read.items():
                if key not in context or context[key] is None:
                    continue
                value = context[key]
                if arg in json_args and not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                kwargs[arg] = value
            error = None
            try:
                frozen = frozen_or_execute(tool_name)
                data = entry.fn(**kwargs) if frozen is _EXECUTE else _replay_data(frozen)
                ok = True
            except Exception as exc:
                data = {"message": str(exc)}
                ok = False
                error = str(exc)
            published = {}
            if ok and isinstance(data, dict):
                for ctx_key, result_key in write.items():
                    if result_key in data:
                        published[ctx_key] = data[result_key]
            context.update(published)
            output = dict(state.get("output") or {})
            output.update(published)
            span = new_span(
                kind="tool",
                name=tool_name,
                span_input=kwargs,
                span_output=data,
                started=started,
                error=error,
                error_kind="tool" if error else None,
                state_before=state.get("shared_context"),
                state_after=context,
            )
            errors = []
            if error:
                errors.append({"source": tool_name, "message": error, "error_kind": "tool"})
            return {"shared_context": context, "output": output, "trace": [span], "errors": errors}

        builder.add_node(node_name, node)


class BaseSubgraph:
    """固定流程。默认按 steps 顺序串行；复杂流程覆盖 build。"""

    name = ""
    description = ""
    writes: list[str] = []
    reads: list[str] = []
    steps: list[str] = []

    def build(self, builder: StateGraph, runtime: SubgraphRuntime) -> None:
        last = runtime.add_sequence(builder, list(self.steps))
        builder.add_edge(last, END)

    def compile(self, agents: dict[str, BaseAgent], tools: dict[str, Any]):
        runtime = SubgraphRuntime(agents, tools)
        builder = StateGraph(SubgraphState)
        self.build(builder, runtime)
        return RunnableSubgraph(self, builder.compile())
