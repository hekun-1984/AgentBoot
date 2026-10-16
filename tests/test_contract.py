import json

from langchain_core.messages import AIMessage

from agentboot import (
    BaseAgent,
    BaseGraph,
    BaseSubgraph,
    JsonlTraceSink,
    MemoryTraceSink,
    ScriptedModel,
    tool,
)
from agentboot.application import AgentBootApplication
from diabetes import build_app


def test_output_type_and_missing_field_follow_failure_policy(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        """---
name: typed
description: 校验输出
tools: []
max_steps: 2
on_failure: return_partial
input: {}
output:
  count:
    type: int
    description: 数量
  label:
    type: str
    description: 标签
---

只输出 JSON。
""",
        encoding="utf-8",
    )

    class Typed(BaseAgent):
        name = "typed"
        skill_path = str(skill)

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"count": "1"}))

    result = Typed().bind({}).invoke({}, user_input="count")
    assert result.error_kind == "llm"
    assert "类型应为 int" in result.error
    assert "缺少字段 label" in result.error


def test_required_tool_blocks_finish_until_it_is_called():
    called = {"value": 0}

    @tool(name="ping", description="回声")
    def ping(value: str) -> dict:
        called["value"] += 1
        return {"value": value}

    class MustPing(BaseAgent):
        name = "must_ping"
        description = "必须先调用 ping"
        tools = ["ping"]
        required_tools = ["ping"]
        max_steps = 3
        output_fields = {"text": "文本"}

        def offline_step(self, messages, shared_context):
            raise AssertionError("应走脚本模型")

    model = ScriptedModel(
        [
            AIMessage(content=json.dumps({"text": "太早"})),
            AIMessage(content="", tool_calls=[{"name": "ping", "args": {"value": "ok"}, "id": "1"}]),
            AIMessage(content=json.dumps({"text": "ok"})),
        ]
    )
    result = MustPing().bind({"ping": ping.__agentboot_tool__}, model).invoke({}, user_input="go")
    assert result.error is None
    assert result.output["text"] == "ok"
    assert called["value"] == 1


def test_subgraph_checks_reads_before_start():
    class Marker(BaseAgent):
        name = "marker"
        description = "标记"
        output_fields = {"token": "标记"}

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"token": shared_context["patient_id"]}))

    class NeedsPatient(BaseSubgraph):
        name = "needs_patient"
        description = "需要患者"
        reads = ["patient_id"]
        writes = ["token"]
        steps = ["marker"]

    class Plain(BaseGraph):
        name = "plain"
        description = "直达"
        subgraphs = ["needs_patient"]
        fallback_subgraph = "needs_patient"

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [{"subgraph": "needs_patient"}], "reason": "固定"}

        def render_answer(self, state):
            return "ok"

    app = AgentBootApplication()
    app.include(Marker, NeedsPatient, Plain)
    blocked = app.invoke_subgraph("needs_patient", {}, user_input="go")
    assert blocked["errors"][0]["error_kind"] == "state"
    assert "patient_id" in blocked["errors"][0]["message"]

    allowed = app.invoke_subgraph("needs_patient", {"patient_id": "p001"}, user_input="go")
    assert allowed["errors"] == []
    assert allowed["shared_context"]["token"] == "p001"


def test_guard_uses_the_field_declared_by_the_graph():
    class Mark(BaseAgent):
        name = "mark"
        description = "标记优先级"
        output_fields = {"priority": "优先级"}

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"priority": "urgent"}))

    class RescueAgent(BaseAgent):
        name = "rescue_agent"
        description = "接手"
        output_fields = {"done": "完成"}

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"done": "yes"}))

    class MarkJob(BaseSubgraph):
        name = "mark_job"
        description = "只标记"
        writes = ["priority"]
        steps = ["mark"]

    class RescueJob(BaseSubgraph):
        name = "rescue"
        description = "兜底子图"
        writes = ["done"]
        steps = ["rescue_agent"]

    class Guarded(BaseGraph):
        name = "guarded"
        description = "按声明兜底"
        subgraphs = ["mark_job", "rescue"]
        fallback_subgraph = "mark_job"
        guard_field = "priority"
        guard_values = ("urgent",)
        guard_subgraph = "rescue"

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [{"subgraph": "mark_job"}], "reason": "先标记"}

        def render_answer(self, state):
            return state["shared_context"].get("done") or ""

    result = AgentBootApplication().include(Mark, RescueAgent, MarkJob, RescueJob, Guarded).invoke("go")
    assert result["answer"] == "yes"
    assert "rescue" in result["results"]
    assert "priority=urgent" in result["route_reason"]


def test_trace_sink_keeps_one_call_and_can_replay(tmp_path):
    memory = MemoryTraceSink()
    app = build_app(trace_sink=memory)
    question = "糖尿病最新指南里，HbA1c 控制目标是多少？"
    context = {"patient_id": "p001"}
    first = app.invoke(question, context)
    assert len(memory.records) == 1
    record = memory.records[0]
    assert record["call_id"] == first["call_id"]
    assert record["user_input"] == question
    assert record["context"] == context

    def walk(spans):
        for span in spans:
            assert span["call_id"] == first["call_id"]
            walk(span.get("children") or [])

    walk(first["trace"])
    kinds = {span["kind"] for span in first["trace"]}
    assert "subgraph" in kinds
    assert "graph" in kinds

    replayed = app.replay(record, trace_sink=MemoryTraceSink())
    assert replayed["answer"] == first["answer"]
    assert replayed["call_id"] != first["call_id"]

    jsonl = JsonlTraceSink(tmp_path / "audit.jsonl")
    app.invoke(question, context, trace_sink=jsonl)
    stored = jsonl.read()
    assert stored[-1]["output"]["answer"] == first["answer"]
    assert stored[-1]["call_id"]


def _typed_skill(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        """---
name: typed
description: 校验输出
tools: []
max_steps: 2
on_failure: return_partial
input: {}
output:
  count:
    type: int
    description: 数量
  label:
    type: str
    description: 标签
---

只输出声明的字段。
""",
        encoding="utf-8",
    )
    return skill


def test_declared_output_is_requested_as_structured_schema(tmp_path):
    skill = _typed_skill(tmp_path)

    class Bound:
        def __init__(self):
            self.free_calls = 0
            self.schemas = []

        def invoke(self, messages, tools):
            self.free_calls += 1
            return AIMessage(content="不是 JSON")

        def invoke_structured(self, messages, schema):
            self.schemas.append(schema)
            return {"count": 4, "label": "done"}

    class Typed(BaseAgent):
        name = "typed"
        skill_path = str(skill)

        def offline_step(self, messages, shared_context):
            raise AssertionError("应走结构化输出")

    model = Bound()
    result = Typed().bind({}, model).invoke({}, user_input="count")
    assert result.error is None
    assert result.output == {"count": 4, "label": "done"}
    assert model.free_calls == 0
    fields = model.schemas[0].model_fields
    assert fields["count"].annotation is int
    assert fields["label"].annotation is str


def test_structured_output_still_rejects_wrong_types(tmp_path):
    skill = _typed_skill(tmp_path)

    class Bound:
        def invoke(self, messages, tools):
            raise AssertionError("没有工具时不应自由生成")

        def invoke_structured(self, messages, schema):
            return {"count": "x", "label": "ok"}

    class Typed(BaseAgent):
        name = "typed"
        skill_path = str(skill)

        def offline_step(self, messages, shared_context):
            raise AssertionError("应走结构化输出")

    result = Typed().bind({}, Bound()).invoke({}, user_input="count")
    assert result.error_kind == "llm"
    assert "类型应为 int" in result.error


def test_required_tool_finishes_through_structured_output():
    @tool(name="ping", description="回声")
    def ping(value: str) -> dict:
        return {"value": value}

    class Bound:
        def __init__(self):
            self.invokes = 0
            self.structured = 0

        def invoke(self, messages, tools):
            self.invokes += 1
            return AIMessage(content="", tool_calls=[{"name": "ping", "args": {"value": "ok"}, "id": "1"}])

        def invoke_structured(self, messages, schema):
            self.structured += 1
            assert "text" in schema.model_fields
            return {"text": "ok"}

    class MustPing(BaseAgent):
        name = "must_ping"
        description = "必须先调用 ping"
        tools = ["ping"]
        required_tools = ["ping"]
        max_steps = 3
        output_fields = {"text": "文本"}

        def offline_step(self, messages, shared_context):
            raise AssertionError("应走脚本或结构化输出")

    model = Bound()
    result = MustPing().bind({"ping": ping.__agentboot_tool__}, model).invoke({}, user_input="go")
    assert result.error is None
    assert result.output["text"] == "ok"
    assert model.invokes == 1
    assert model.structured == 1


def test_optional_tool_turn_is_followed_by_structured_output():
    @tool(name="ping", description="回声")
    def ping(value: str) -> dict:
        return {"value": value}

    class Bound:
        def __init__(self):
            self.invokes = 0
            self.structured = 0

        def invoke(self, messages, tools):
            self.invokes += 1
            return AIMessage(content="自由文本")

        def invoke_structured(self, messages, schema):
            self.structured += 1
            return {"text": "from-schema"}

    class OptionalPing(BaseAgent):
        name = "optional_ping"
        description = "可以调用 ping"
        tools = ["ping"]
        max_steps = 3
        output_fields = {"text": "文本"}

        def offline_step(self, messages, shared_context):
            raise AssertionError("应走结构化输出")

    model = Bound()
    result = OptionalPing().bind({"ping": ping.__agentboot_tool__}, model).invoke({}, user_input="go")
    assert result.error is None
    assert result.output["text"] == "from-schema"
    assert model.invokes == 1
    assert model.structured == 1


def test_langchain_adapter_calls_with_structured_output():
    from agentboot.contract import output_model
    from agentboot.llm import LangChainModel
    from agentboot.models import FieldSpec

    class Chat:
        def __init__(self):
            self.schema = None

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, messages):
            return self.schema(count=2, label="ok")

    chat = Chat()
    schema = output_model(
        "typed",
        {
            "count": FieldSpec("count", "int", "数量", required=True),
            "label": FieldSpec("label", "str", "标签", required=True),
        },
    )
    data = LangChainModel(chat).invoke_structured([], schema)
    assert chat.schema is schema
    assert data == {"count": 2, "label": "ok"}


def test_distribution_contains_only_the_framework():
    import tomllib
    from pathlib import Path

    data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["setuptools"]["packages"]["find"]["include"] == ["agentboot*"]


def test_router_requests_a_plan_limited_to_registered_subgraphs():
    class StructuredRouter:
        def __init__(self):
            self.schema = None

        def invoke(self, messages, tools):
            raise AssertionError("应走结构化路由")

        def invoke_structured(self, messages, schema):
            self.schema = schema
            return {"reason": "按表选择", "steps": [{"subgraph": "only", "parallel_group": None}]}

    class Marker(BaseAgent):
        name = "marker"
        description = "标记"
        output_fields = {"token": "标记"}

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"token": "yes"}))

    class Only(BaseSubgraph):
        name = "only"
        description = "唯一子图"
        writes = ["token"]
        steps = ["marker"]

    class Box(BaseGraph):
        name = "box"
        description = "主图"
        subgraphs = ["only", "other"]
        fallback_subgraph = "other"

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [], "reason": "不应走离线"}

        def render_answer(self, state):
            return str((state.get("shared_context") or {}).get("token") or "")

    class OtherAgent(BaseAgent):
        name = "other_agent"
        description = "兜底"
        output_fields = {"token": "标记"}

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"token": "fallback"}))

    class Other(BaseSubgraph):
        name = "other"
        description = "兜底子图"
        writes = ["token"]
        steps = ["other_agent"]

    router = StructuredRouter()
    result = (
        AgentBootApplication(router=router)
        .include(Marker, OtherAgent, Only, Other, Box)
        .invoke("go")
    )
    schema = router.schema.model_json_schema()
    assert "only" in json.dumps(schema, ensure_ascii=False)
    assert "other" in json.dumps(schema, ensure_ascii=False)
    assert result["answer"] == "yes"
    assert result["plan"]["steps"][0]["subgraph"] == "only"


def test_frozen_replay_does_not_call_the_tool_again():
    from agentboot.exceptions import ToolReplayError
    from agentboot.tool import tool_replay

    calls = {"n": 0}

    @tool(name="ping", description="回声")
    def ping(value: str) -> dict:
        calls["n"] += 1
        return {"value": value}

    recorded = json.dumps({"ok": True, "data": {"value": "old"}}, ensure_ascii=False)
    record = {"trace": [{"kind": "tool", "name": "ping", "output": recorded}]}
    with tool_replay(record):
        assert ping.__agentboot_tool__.lc_tool.invoke({"value": "new"}) == recorded
    assert calls["n"] == 0
    with tool_replay({"trace": []}):
        try:
            ping.__agentboot_tool__.lc_tool.invoke({"value": "new"})
        except ToolReplayError:
            pass
        else:
            raise AssertionError("记录里没有的调用不应执行")
    assert calls["n"] == 0
