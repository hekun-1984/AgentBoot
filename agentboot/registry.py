"""组件表。名字是白名单，重复注册会失败。"""

from __future__ import annotations

from agentboot.exceptions import RegistrationError
from agentboot.tool import ToolEntry


class Registry:
    def __init__(self) -> None:
        self.tools: dict[str, ToolEntry] = {}
        self.agents: dict[str, type] = {}
        self.subgraphs: dict[str, type] = {}
        self.graphs: dict[str, type] = {}

    def add_tool(self, entry: ToolEntry) -> None:
        current = self.tools.get(entry.name)
        if current is not None and current.fn is not entry.fn:
            raise RegistrationError(f"Tool 重名: {entry.name}")
        self.tools[entry.name] = entry

    def add_agent(self, cls: type) -> None:
        name = getattr(cls, "name", "")
        if not name:
            raise RegistrationError(f"{cls.__name__} 缺少 name")
        current = self.agents.get(name)
        if current is not None and current is not cls:
            raise RegistrationError(f"Agent 重名: {name}")
        self.agents[name] = cls

    def add_subgraph(self, cls: type) -> None:
        name = getattr(cls, "name", "")
        if not name:
            raise RegistrationError(f"{cls.__name__} 缺少 name")
        current = self.subgraphs.get(name)
        if current is not None and current is not cls:
            raise RegistrationError(f"Subgraph 重名: {name}")
        self.subgraphs[name] = cls

    def add_graph(self, cls: type) -> None:
        name = getattr(cls, "name", "")
        if not name:
            raise RegistrationError(f"{cls.__name__} 缺少 name")
        current = self.graphs.get(name)
        if current is not None and current is not cls:
            raise RegistrationError(f"Graph 重名: {name}")
        self.graphs[name] = cls

    def add(self, component: object) -> None:
        entry = getattr(component, "__agentboot_tool__", None)
        if entry is not None:
            self.add_tool(entry)
            return
        if isinstance(component, type):
            from agentboot.agent import BaseAgent
            from agentboot.graph import BaseGraph
            from agentboot.subgraph import BaseSubgraph

            if issubclass(component, BaseGraph):
                self.add_graph(component)
                return
            if issubclass(component, BaseSubgraph):
                self.add_subgraph(component)
                return
            if issubclass(component, BaseAgent):
                self.add_agent(component)
                return
        raise RegistrationError(f"无法注册 {component!r}")
