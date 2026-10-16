import pytest
from langchain_core.exceptions import (
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.messages import AIMessage, HumanMessage

from agentboot import BaseAgent, BaseGraph, BaseSubgraph, ModelCatalog, ScriptedModel
from agentboot.application import AgentBootApplication
from agentboot.catalog import EmbeddingClient, ModelOption, RerankClient, build_reranker, resolve_api_key
from agentboot.exceptions import ConfigurationError
from agentboot.llm import LangChainModel, is_unavailable
from agentboot.skill import load_skill
from agentboot.tracing import message_brief


def _route_config():
    return {
        "route": {
            "default": "fast-a",
            "options": [
                {
                    "name": "fast-b",
                    "provider": "dashscope",
                    "model": "qwen-turbo",
                    "api_key_env": "DASHSCOPE_API_KEY",
                },
                {
                    "name": "fast-a",
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "api_key_env": "OPENAI_API_KEY",
                },
            ],
        },
        "embed": {
            "default": "vectors",
            "options": [
                {"name": "vectors", "provider": "openai", "model": "text-embedding-3-small", "api_key_env": "OPENAI_API_KEY"},
            ],
        },
        "rerank": {
            "options": [
                {"name": "ranker", "provider": "cohere", "model": "rerank-v3.5", "api_key_env": "COHERE_API_KEY"},
            ],
        },
    }


def test_catalog_puts_the_default_model_first():
    catalog = ModelCatalog.parse(_route_config())
    assert [item.name for item in catalog.slots["route"].ordered()] == ["fast-a", "fast-b"]
    assert catalog.slots["embed"].kind == "embed"
    assert catalog.slots["rerank"].kind == "rerank"
    assert catalog.slots["rerank"].default == "ranker"


def test_catalog_rejects_ambiguous_or_secret_config():
    with pytest.raises(ConfigurationError, match="default"):
        ModelCatalog.parse(
            {
                "general": {
                    "options": [
                        {"name": "a", "provider": "openai", "model": "gpt-4.1-mini"},
                        {"name": "b", "provider": "openai", "model": "gpt-4.1"},
                    ]
                }
            }
        )
    with pytest.raises(ConfigurationError, match="不在名单"):
        ModelCatalog.parse(
            {
                "general": {
                    "default": "missing",
                    "options": [{"name": "a", "provider": "openai", "model": "gpt-4.1-mini"}],
                }
            }
        )
    with pytest.raises(ConfigurationError, match="重名"):
        ModelCatalog.parse(
            {
                "general": {
                    "default": "a",
                    "options": [
                        {"name": "a", "provider": "openai", "model": "gpt-4.1-mini"},
                        {"name": "a", "provider": "dashscope", "model": "qwen-plus"},
                    ],
                }
            }
        )
    with pytest.raises(ConfigurationError, match="api_key_env"):
        ModelCatalog.parse(
            {
                "general": {
                    "options": [{"name": "a", "provider": "openai", "model": "gpt-4.1-mini", "api_key": "sk-secret"}],
                }
            }
        )


def test_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    option = ModelOption(name="fast-a", provider="openai", model="gpt-4.1-mini", api_key_env="OPENAI_API_KEY")
    assert resolve_api_key(option) == "sk-from-env"
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        ModelCatalog.parse(_route_config()).chat("route")


def test_env_file_loads_the_catalog(monkeypatch, tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        """
general:
  default: main
  options:
    - name: main
      provider: openai
      model: gpt-4.1-mini
      api_key_env: OPENAI_API_KEY
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTBOOT_MODELS", str(path))
    app = AgentBootApplication()
    assert app.catalog.slots["general"].default == "main"


def test_unavailable_model_switches_to_the_backup_and_records_its_name():
    class Down:
        def bind_tools(self, tools):
            self.tools = tools
            return self

        def invoke(self, messages):
            raise ModelConnectionError("down")

    class Up:
        def bind_tools(self, tools):
            self.tools = tools
            return self

        def invoke(self, messages):
            return AIMessage(content="ok")

    tool = object()
    model = LangChainModel(candidates=[("fast-a", Down()), ("fast-b", Up())])
    message = model.invoke([HumanMessage(content="hi")], [tool])
    assert message.content == "ok"
    assert model.last_option == "fast-b"
    assert message_brief(message)["model_option"] == "fast-b"


def test_bad_model_output_does_not_switch():
    class Bad:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            raise OutputParserException("不是声明的结构")

    class Later:
        def __init__(self):
            self.called = False

        def with_structured_output(self, schema):
            self.called = True
            return self

        def invoke(self, messages):
            return {"text": "no"}

    later = Later()
    model = LangChainModel(candidates=[("fast-a", Bad()), ("fast-b", later)])
    with pytest.raises(OutputParserException):
        model.invoke_structured([], object())
    assert later.called is False

    class Invalid:
        def invoke(self, messages):
            raise ModelInvalidRequestError("请求不合法")

    assert is_unavailable(ModelTimeoutError("slow")) is True
    assert is_unavailable(ModelInvalidRequestError("bad")) is False

    class Status(Exception):
        status_code = 503

    assert is_unavailable(Status()) is True
    with pytest.raises(ModelInvalidRequestError):
        LangChainModel(candidates=[("fast-a", Invalid()), ("fast-b", Later())]).invoke([], [])


def test_embedding_and_rerank_use_their_own_backup():
    class Down:
        def embed_documents(self, texts):
            raise ModelTimeoutError("slow")

        def rerank(self, query, documents):
            raise ModelConnectionError("down")

    class Up:
        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

        def rerank(self, query, documents):
            return [{"index": 1, "relevance_score": 0.8}]

    vectors = EmbeddingClient([("main", Down()), ("backup", Up())])
    assert vectors.embed(["票据"]) == [[0.1, 0.2]]
    assert vectors.last_option == "backup"

    ranked = RerankClient([("main", Down()), ("backup", Up())])
    assert ranked.rerank("限额", ["甲", "乙"]) == [{"index": 1, "score": 0.8}]
    assert ranked.last_option == "backup"

    with pytest.raises(ConfigurationError, match="对话模型"):
        ModelCatalog.parse(_route_config()).chat("embed")
    option = ModelOption(name="ranker", provider="missingvendor", model="rerank-v3.5")
    with pytest.raises(ConfigurationError, match="langchain-missingvendor"):
        build_reranker(option)


def test_agents_and_router_receive_the_declared_slot():
    reason = ScriptedModel([AIMessage(content='{"text":"deep"}')])
    general = ScriptedModel([AIMessage(content='{"text":"plain"}')])
    route = ScriptedModel([AIMessage(content='{"steps":[],"reason":"route"}')])
    catalog = ModelCatalog().bind("reason", reason).bind("general", general).bind("route", route)

    class ReasonAgent(BaseAgent):
        name = "reason_agent"
        description = "强推理"
        model_slot = "reason"
        output_fields = {"text": "文本"}

        def offline_step(self, messages, shared_context):
            raise AssertionError("应使用 reason 槽位")

    class PlainAgent(BaseAgent):
        name = "plain_agent"
        description = "一般文字"
        output_fields = {"text": "文本"}

        def offline_step(self, messages, shared_context):
            raise AssertionError("应使用 general 槽位")

    class One(BaseSubgraph):
        name = "one"
        description = "一步"
        writes = ["text"]
        steps = ["plain_agent"]

    class Box(BaseGraph):
        name = "box"
        description = "主图"
        subgraphs = ["one"]
        fallback_subgraph = "one"

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [{"subgraph": "one"}], "reason": "固定"}

        def render_answer(self, state):
            return "ok"

    app = AgentBootApplication(models=catalog)
    app.include(ReasonAgent, PlainAgent, One, Box)
    compiled = app.compile()
    assert compiled.agents["reason_agent"].model is reason
    assert compiled.agents["plain_agent"].model is general
    assert compiled.graph.model is route

    explicit = ScriptedModel([])
    routed = AgentBootApplication(models=catalog, router=explicit)
    routed.include(ReasonAgent, PlainAgent, One, Box)
    assert routed.compile().graph.model is explicit

    fallback = ModelCatalog().bind("general", general)
    held = AgentBootApplication(models=fallback)
    held.include(PlainAgent, One, Box)
    assert held.compile().graph.model is general


def test_missing_declared_slot_fails_at_compile():
    class VisionAgent(BaseAgent):
        name = "vision_agent"
        description = "看图"
        model_slot = "vision"
        output_fields = {"text": "文本"}

        def offline_step(self, messages, shared_context):
            return AIMessage(content='{"text":"offline"}')

    class One(BaseSubgraph):
        name = "one"
        description = "一步"
        writes = ["text"]
        steps = ["vision_agent"]

    class Box(BaseGraph):
        name = "box"
        description = "主图"
        subgraphs = ["one"]
        fallback_subgraph = "one"

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [{"subgraph": "one"}], "reason": "固定"}

        def render_answer(self, state):
            return "ok"

    app = AgentBootApplication(models=ModelCatalog().bind("general", ScriptedModel([])))
    app.include(VisionAgent, One, Box)
    with pytest.raises(ConfigurationError, match="vision"):
        app.compile()


def test_skill_model_field_selects_the_slot(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        """---
name: reader
description: 抽取
tools: []
max_steps: 2
on_failure: return_partial
model: extract
input: {}
output:
  text:
    type: str
    description: 文本
---

抽取字段。
""",
        encoding="utf-8",
    )
    assert load_skill(skill).model_slot == "extract"


def test_shipped_model_examples_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples"
    shared = ModelCatalog.read(root / "models.yaml")
    diabetes = ModelCatalog.read(root / "diabetes" / "models.yaml")
    expense = ModelCatalog.read(root / "expense" / "models.yaml")
    assert set(shared.slots) == {"route", "general", "extract", "reason", "vision", "embed", "rerank"}
    assert shared.slots["route"].default == "fast-a"
    assert [item.name for item in shared.slots["route"].ordered()] == ["fast-a", "fast-b"]
    assert diabetes.slots["reason"].default == "deep"
    assert "embed" in diabetes.slots and "rerank" in diabetes.slots
    assert expense.slots["reason"].kind == "chat"
    assert expense.slots["rerank"].kind == "rerank"
