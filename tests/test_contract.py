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
