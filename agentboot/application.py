"""应用入口。扫描业务包，把四层装成一张可调用的主图。"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from pathlib import Path
from typing import Any

from agentboot.agent import BaseAgent
from agentboot.audit import TraceSink, build_record, call_scope, new_call_id
from agentboot.exceptions import RegistrationError
from agentboot.graph import BaseGraph
from agentboot.catalog import ModelCatalog, use_catalog
from agentboot.llm import ChatModel, load_model
from agentboot.registry import Registry
from agentboot.subgraph import BaseSubgraph
from agentboot.tool import ToolEntry


class CompiledApplication:
    def __init__(
        self,
        graph: BaseGraph,
        agents: dict[str, BaseAgent],
        subgraphs: dict[str, Any],
        trace_sink: TraceSink | None = None,
    ) -> None:
        self.graph = graph
        self.agents = agents
        self.subgraphs = subgraphs
        self.compiled = graph.graph
        self.trace_sink = trace_sink

    def invoke(self, user_input: str, context: dict | None = None, *, trace_sink: TraceSink | None = None) -> dict:
        initial = dict(context or {})
        shared = dict(initial)
        shared.setdefault("user_input", user_input)
        call_id = new_call_id()
        with call_scope(call_id):
            state = self.compiled.invoke(
                {
                    "user_input": user_input,
                    "shared_context": shared,
                    "results": {},
                    "trace": [],
                    "errors": [],
                    "wave_index": 0,
                    "step_count": 0,
                }
            )
        final = dict(state.get("final_output") or {})
        final["call_id"] = call_id
        record = build_record(call_id=call_id, user_input=user_input, context=initial, output=final)
        sink = self.trace_sink if trace_sink is None else trace_sink
        if sink is not None:
            sink.write(record)
        return final

    def replay(self, record: dict, *, trace_sink: TraceSink | None = None) -> dict:
        return self.invoke(str(record.get("user_input") or ""), dict(record.get("context") or {}), trace_sink=trace_sink)

    def invoke_agent(self, name: str, context: dict | None = None, *, user_input: str = "") -> Any:
        if name not in self.agents:
            raise RegistrationError(f"没有名为 {name} 的 Agent")
        payload = dict(context or {})
        text = user_input or str(payload.get("user_input") or "")
        return self.agents[name].invoke(payload, user_input=text)

    def invoke_subgraph(self, name: str, context: dict | None = None, *, user_input: str = "") -> dict:
        if name not in self.subgraphs:
            raise RegistrationError(f"没有名为 {name} 的 Subgraph")
        payload = dict(context or {})
        text = user_input or str(payload.get("user_input") or "")
        payload.setdefault("user_input", text)
        return self.subgraphs[name].invoke(
            {"user_input": text, "shared_context": payload, "output": {}, "trace": [], "errors": []}
        )


_UNSET = object()


def _resolve_model(model: ChatModel | str | None) -> ChatModel | None:
    if model is None or model == "":
        return None
    if isinstance(model, str):
        return load_model(model)
    return model


class AgentBootApplication:
    """类似 Spring Boot 的装配器：scan 一个业务包，invoke 一次请求。"""

    def __init__(
        self,
        *,
        model: ChatModel | str | None = None,
        router: Any = _UNSET,
        graph: str | None = None,
        trace_sink: TraceSink | None = None,
        models: ModelCatalog | str | Path | dict | None = None,
    ) -> None:
        self.registry = Registry()
        self.graph_name = graph
        self.model = _resolve_model(model)
        self._router_set = router is not _UNSET
        self.router = self.model if router is _UNSET else _resolve_model(router)
        self.catalog = _resolve_catalog(models)
        use_catalog(self.catalog)
        self.trace_sink = trace_sink
        self._compiled: CompiledApplication | None = None

    def include(self, *components: object) -> "AgentBootApplication":
        for component in components:
            self.registry.add(component)
        self._compiled = None
        return self

    def scan(self, package_name: str) -> "AgentBootApplication":
        package = importlib.import_module(package_name)
        modules = [package]
        if hasattr(package, "__path__"):
            for found in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                modules.append(importlib.import_module(found.name))
        for module in modules:
            for _, obj in inspect.getmembers(module):
                entry = getattr(obj, "__agentboot_tool__", None)
                if isinstance(entry, ToolEntry):
                    self.registry.add_tool(entry)
                if not inspect.isclass(obj) or obj.__module__ != module.__name__:
                    continue
                if obj in (BaseAgent, BaseSubgraph, BaseGraph):
                    continue
                if issubclass(obj, BaseGraph):
                    self.registry.add_graph(obj)
                elif issubclass(obj, BaseSubgraph):
                    self.registry.add_subgraph(obj)
                elif issubclass(obj, BaseAgent):
                    self.registry.add_agent(obj)
        self._compiled = None
        return self

    def compile(self) -> CompiledApplication:
        if self._compiled is not None:
            return self._compiled
        if not self.registry.graphs:
            raise RegistrationError("至少要注册一个主图")
        if self.graph_name:
            graph_cls = self.registry.graphs[self.graph_name]
        elif len(self.registry.graphs) == 1:
            graph_cls = next(iter(self.registry.graphs.values()))
        else:
            names = ", ".join(self.registry.graphs)
            raise RegistrationError(f"有多张主图，请指定 graph。当前: {names}")

        agents = {}
        for name, cls in self.registry.agents.items():
            agent = cls()
            agents[name] = agent.bind(self.registry.tools, self._model_for(agent.spec.model_slot or "general"))
        subgraphs = {}
        for name, cls in self.registry.subgraphs.items():
            subgraphs[name] = cls().compile(agents, self.registry.tools)
        graph = graph_cls().bind(subgraphs, self.registry.tools, self._router_model())
        self._compiled = CompiledApplication(graph, agents, subgraphs, self.trace_sink)
        return self._compiled

    def invoke(self, user_input: str, context: dict | None = None, *, trace_sink: TraceSink | None = None) -> dict:
        return self.compile().invoke(user_input, context, trace_sink=trace_sink)

    def replay(self, record: dict, *, trace_sink: TraceSink | None = None) -> dict:
        return self.compile().replay(record, trace_sink=trace_sink)

    def invoke_agent(self, name: str, context: dict | None = None, *, user_input: str = ""):
        return self.compile().invoke_agent(name, context, user_input=user_input)

    def invoke_subgraph(self, name: str, context: dict | None = None, *, user_input: str = "") -> dict:
        return self.compile().invoke_subgraph(name, context, user_input=user_input)

    def _model_for(self, slot: str) -> Any:
        if self.catalog is None:
            return self.model
        return self.catalog.chat(slot)

    def _router_model(self) -> Any:
        if self._router_set:
            return self.router
        if self.catalog is None:
            return self.model
        if self.catalog.has("route"):
            return self.catalog.chat("route")
        if self.catalog.has("general"):
            return self.catalog.chat("general")
        return self.model


def _resolve_catalog(models: ModelCatalog | str | Path | dict | None) -> ModelCatalog | None:
    if isinstance(models, ModelCatalog):
        return models
    if isinstance(models, dict):
        return ModelCatalog.parse(models)
    if isinstance(models, (str, Path)) and str(models).strip():
        return ModelCatalog.read(models)
    if models is not None:
        return None
    path = os.getenv("AGENTBOOT_MODELS")
    if path:
        return ModelCatalog.read(path)
    return None
