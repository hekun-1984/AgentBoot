"""白名单路由控制器。防止 LLM 自由调用任意能力。"""

from __future__ import annotations

from typing import Any

from agentboot.config import Settings
from agentboot.exceptions import RoutingError
from agentboot.logger import get_logger

logger = get_logger(__name__)

_PERMISSIVE = {"allowed_skills": ["*"]}


class Router:
    """白名单路由器。路由表：agent_name -> {allowed_skills, max_steps, timeout}。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._routes: dict[str, dict[str, Any]] = {}

    def load_whitelist(self) -> None:
        """从 routes/route_config.py 加载白名单配置。"""
        try:
            from routes.route_config import ROUTE_TABLE
        except ImportError:
            logger.warning("No route_config.py found, using permissive mode.")
            self._routes = {"_default": dict(_PERMISSIVE)}
            return
        self.load_table(ROUTE_TABLE)

    def load_table(self, table: dict[str, dict[str, Any]]) -> None:
        self._routes = table
        logger.info("Loaded %d route configurations.", len(self._routes))

    def check_whitelist(self, agent_name: str, requested_skills: list[str]) -> bool:
        """检查请求的能力是否在白名单内。"""
        route = self.get_route(agent_name)
        allowed = route.get("allowed_skills", ["*"])
        if "*" in allowed:
            return True
        denied = [skill for skill in requested_skills if skill not in allowed]
        if denied:
            raise RoutingError(
                f"Agent '{agent_name}' denied skills: {denied}. Allowed: {allowed}."
            )
        return True

    def get_route(self, agent_name: str) -> dict[str, Any]:
        if agent_name in self._routes:
            return self._routes[agent_name]
        return self._routes.get("_default", dict(_PERMISSIVE))
