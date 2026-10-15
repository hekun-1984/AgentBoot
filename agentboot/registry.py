"""子图/能力注册中心。类似 Spring 的 BeanFactory，统一管理 Agent、SkillGraph 和 Capability。"""

from __future__ import annotations

from typing import Any

from agentboot.exceptions import RegistrationError
from agentboot.graph import BaseSubGraph
from agentboot.logger import get_logger

logger = get_logger(__name__)


def _logical_name(component: type, fallback: str) -> str:
    name = getattr(component, "name", None)
    if isinstance(name, str) and name:
        return name
    return fallback


class Registry:
    """全局注册中心。注册 Agent 类、能力类、子图类。"""

    def __init__(self) -> None:
        self._agents: dict[str, type] = {}
        self._agent_instances: dict[str, Any] = {}
        self._aliases: dict[str, str] = {}
        self._skills: dict[str, type] = {}
        self._subgraphs: dict[str, type[BaseSubGraph]] = {}

    def register(self, agent_class: type) -> Any:
        if not isinstance(agent_class, type):
            raise RegistrationError("register() expects an agent class.")
        instance = agent_class()
        name = _logical_name(agent_class, agent_class.__name__)
        if name in self._agents:
            raise RegistrationError(f"Agent '{name}' already registered.")
        class_name = agent_class.__name__
        if class_name != name and (class_name in self._agents or class_name in self._aliases):
            raise RegistrationError(f"Agent alias '{class_name}' already registered.")
        self._agents[name] = agent_class
        self._agent_instances[name] = instance
        if class_name != name:
            self._aliases[class_name] = name
        for subgraph_name, graph_class in instance.subgraphs.items():
            self.register_subgraph(subgraph_name, graph_class)
        for skill_class in instance.skills.values():
            self.register_skill(skill_class)
        logger.info("Registered agent %s with %d subgraphs.", name, len(instance.subgraphs))
        return instance

    def register_skill(self, skill_class: type) -> None:
        name = _logical_name(skill_class, skill_class.__name__)
        existing = self._skills.get(name)
        if existing is not None and existing is not skill_class:
            raise RegistrationError(f"Skill '{name}' already registered.")
        if existing is skill_class:
            return
        instance = skill_class()
        errors = instance.validate() if hasattr(instance, "validate") else []
        if errors:
            raise RegistrationError("; ".join(errors))
        self._skills[name] = skill_class

    def register_subgraph(self, name: str, graph_class: type[BaseSubGraph]) -> None:
        existing = self._subgraphs.get(name)
        if existing is not None and existing is not graph_class:
            raise RegistrationError(f"SubGraph '{name}' already registered.")
        if existing is graph_class:
            return
        self._subgraphs[name] = graph_class

    def resolve_agent_name(self, name: str) -> str | None:
        if name in self._agents:
            return name
        return self._aliases.get(name)

    def get_agent(self, name: str) -> type | None:
        resolved = self.resolve_agent_name(name)
        if resolved is None:
            return None
        return self._agents.get(resolved)

    def get_agent_instance(self, name: str) -> Any | None:
        resolved = self.resolve_agent_name(name)
        if resolved is None:
            return None
        return self._agent_instances.get(resolved)

    def get_skill(self, name: str) -> type | None:
        return self._skills.get(name)

    def get_subgraph(self, name: str) -> type[BaseSubGraph] | None:
        return self._subgraphs.get(name)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def list_subgraphs(self) -> list[str]:
        return list(self._subgraphs.keys())
