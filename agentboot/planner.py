"""LLM 执行计划生成器。有意图且配置了模型时生成计划，失败时回退到 Agent 模板。"""

from __future__ import annotations

import asyncio
import importlib
import json
import pkgutil
import re
import urllib.error
import urllib.request
from typing import Any, Protocol

from agentboot.config import Settings
from agentboot.exceptions import ConfigurationError, ExecutionError
from agentboot.logger import get_logger
from agentboot.models import ExecutionPlan, Step
from agentboot.registry import Registry

logger = get_logger(__name__)


class LLMClient(Protocol):
    async def complete(self, prompt: str) -> str:
        """返回模型原文。"""


def discover_plans() -> dict[str, type]:
    """扫描 plans 包，按 agent 名称收集计划模板。"""
    try:
        import plans
        from plans.base_plan import BasePlan
    except ImportError:
        logger.warning("plans package is not available.")
        return {}
    found: dict[str, type] = {}
    for module_info in pkgutil.iter_modules(plans.__path__):
        if module_info.name.startswith("_") or module_info.name == "base_plan":
            continue
        module = importlib.import_module(f"plans.{module_info.name}")
        for obj in vars(module).values():
            if not isinstance(obj, type) or not issubclass(obj, BasePlan) or obj is BasePlan:
                continue
            agent_name = getattr(obj, "agent", None)
            if not agent_name:
                continue
            found[str(agent_name)] = obj
    logger.info("Discovered %d plan templates.", len(found))
    return found


def parse_plan_payload(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象，兼容 ```json 代码块。"""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM plan must be a JSON object.")
    return payload


def build_llm_client(settings: Settings) -> LLMClient | None:
    if not settings.llm_api_key:
        return None
    return OpenAICompatibleClient(settings)


class OpenAICompatibleClient:
    """最小 OpenAI Chat Completions 客户端，不额外引入 SDK。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete(self, prompt: str) -> str:
        return await asyncio.to_thread(self._post, prompt)

    def _post(self, prompt: str) -> str:
        if self.settings.llm_provider != "openai-compatible":
            raise ConfigurationError(
                f"Unsupported LLM provider '{self.settings.llm_provider}'."
            )
        url = f"{self.settings.llm_base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.settings.llm_model,
                "temperature": self.settings.llm_temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.llm_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ExecutionError(f"LLM request failed: {exc}") from exc
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ExecutionError("LLM response did not contain a message.") from exc


class Planner:
    """根据用户意图生成 ExecutionPlan，或回退到 Agent.execute_plan。"""

    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.llm_client = build_llm_client(settings) if llm_client is None else llm_client
        self.templates: dict[str, type] = {}

    async def plan(self, agent: Any, context: dict[str, Any]) -> ExecutionPlan:
        if context.get("intent") and self.llm_client is not None:
            try:
                return await self._from_llm(agent, context)
            except Exception as exc:
                logger.warning("LLM plan failed, fallback to agent plan: %s", exc)
        return await agent.execute_plan(context)

    async def _from_llm(self, agent: Any, context: dict[str, Any]) -> ExecutionPlan:
        assert self.llm_client is not None
        prompt = self._prompt(agent, context)
        text = await self.llm_client.complete(prompt)
        payload = parse_plan_payload(text)
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("LLM plan has no steps.")
        steps: list[Step] = []
        for item in raw_steps:
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError("LLM step is missing id.")
            steps.append(
                Step(
                    id=str(item["id"]),
                    type=str(item.get("type", "subgraph")),
                    depends_on=[str(dep) for dep in item.get("depends_on", [])],
                    parallel_group=item.get("parallel_group"),
                    config=dict(item.get("config") or {}),
                    timeout=item.get("timeout"),
                )
            )
        plan_id = str(payload.get("plan_id") or f"{agent.name}_llm")
        return ExecutionPlan(
            plan_id=plan_id,
            agent=agent.name,
            steps=steps,
            metadata={"source": "llm", "model": self.settings.llm_model},
        )

    def _prompt(self, agent: Any, context: dict[str, Any]) -> str:
        catalog = []
        for name, graph_class in agent.subgraphs.items():
            catalog.append(
                {
                    "id": name,
                    "type": "subgraph",
                    "description": getattr(graph_class, "description", ""),
                    "depends_on": list(getattr(graph_class, "depends_on", []) or []),
                    "parallel_group": getattr(graph_class, "parallel_group", None),
                    "allowed_skills": list(getattr(graph_class, "allowed_skills", []) or []),
                }
            )
        safe_context = {
            key: value
            for key, value in context.items()
            if key != "intent"
        }
        return (
            "You are the planner of an agent orchestration framework. "
            "Return one JSON object only. Do not invent step ids.\n"
            f"Agent: {agent.name}\n"
            f"Description: {getattr(agent, 'description', '')}\n"
            f"Intent: {context.get('intent')}\n"
            f"Context: {json.dumps(safe_context, ensure_ascii=False, default=str)}\n"
            f"Allowed steps: {json.dumps(catalog, ensure_ascii=False)}\n"
            "Schema: {\"plan_id\": string, \"steps\": [{\"id\": string, \"type\": \"subgraph\", "
            "\"depends_on\": [string], \"parallel_group\": string|null}]}"
        )
