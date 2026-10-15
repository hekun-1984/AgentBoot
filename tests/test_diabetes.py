import json

from langchain_core.messages import AIMessage

from agentboot import BaseAgent, BaseGraph, BaseSubgraph, ScriptedModel
from agentboot.application import AgentBootApplication
from diabetes import build_app
from diabetes.tool.store import STORE


def test_subgraph_keeps_a_fixed_order_and_passes_context():
    class First(BaseAgent):
        name = "first"
        description = "第一步"
        output_fields = {"token": "标记"}

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"token": "A"}))

    class Second(BaseAgent):
        name = "second"
        description = "第二步"
        output_fields = {"seen": "看见的标记"}

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"seen": shared_context.get("token")}))

    class Pipe(BaseSubgraph):
        name = "pipe"
        description = "固定两步"
        writes = ["token", "seen"]
        steps = ["first", "second"]

    class OnlyPipe(BaseGraph):
        name = "only_pipe"
        description = "只跑管道"
        subgraphs = ["pipe"]
        fallback_subgraph = "pipe"
        max_steps = 2

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [{"subgraph": "pipe"}], "reason": "固定"}

        def render_answer(self, state):
            return state["shared_context"].get("seen")

    app = AgentBootApplication()
    app.include(First, Second, Pipe, OnlyPipe)
    result = app.invoke("go")
    assert result["answer"] == "A"
    subgraph_spans = [span for span in result["trace"] if span["kind"] == "subgraph"]
    agent_names = [child["name"] for child in subgraph_spans[0]["children"] if child["kind"] == "agent"]
    assert agent_names == ["first", "second"]


def test_three_diabetes_questions_follow_the_four_layers():
    app = build_app()
    diet = app.invoke("我最近空腹血糖经常在 7.8 左右，饮食和运动该怎么调整？", {"patient_id": "p001"})
    assert diet["plan"]["waves"] == [["diabetes_risk_check", "diabetes_management_plan"]]
    assert diet["shared_context"]["risk_level"] == "medium"
    assert "饮食：" in diet["answer"]
    assert "运动：" in diet["answer"]
    assert "复查：" in diet["answer"]
    assert "7.8" in diet["answer"]
    assert "emergency_escalation" not in diet["results"]
    assert STORE.reminders[-1]["sent"] is True

    emergency = app.invoke("我现在头晕、手抖，刚测血糖 2.9。", {"patient_id": "p001"})
    assert emergency["plan"]["steps"] == [{"subgraph": "emergency_escalation", "parallel_group": None}]
    assert emergency["shared_context"]["risk_level"] == "critical"
    assert "15 克" in emergency["answer"]
    assert "不要自行加减" in emergency["answer"]
    assert "饮食：" not in emergency["answer"]
    assert STORE.events[-1]["risk_type"] == "critical"

    guide = app.invoke("糖尿病最新指南里，HbA1c 控制目标是多少？", {"patient_id": "p001"})
    assert guide["plan"]["steps"][0]["subgraph"] == "diabetes_guideline_qa"
    assert "7.0%" in guide["answer"]
    assert "来源：" in guide["answer"]
    assert guide["shared_context"]["compliance_passed"] is True


def test_record_critical_value_forces_emergency_after_a_wrong_plan():
    result = build_app().invoke("看看我的血糖并给饮食建议", {"patient_id": "p002"})
    assert "emergency_escalation" in result["results"]
    assert result["shared_context"]["risk_level"] == "critical"
    assert "饮食：" not in result["answer"]
    assert "不要自行加减" in result["answer"]


def test_empty_or_unknown_plan_falls_back_to_guideline_whitelist():
    empty = build_app(router=ScriptedModel([AIMessage(content='{"steps": [], "reason": "空白"}')]))
    missed = empty.invoke("今天上海天气怎么样？", {"patient_id": "p001"})
    assert missed["plan"]["steps"][0]["subgraph"] == "diabetes_guideline_qa"
    assert "空计划兜底" in missed["route_reason"]
    assert missed["answer"] == "不在本能力范围"

    unknown = build_app(
        router=ScriptedModel(
            [AIMessage(content='{"steps": [{"subgraph": "finance_plan"}], "reason": "越权"}')]
        )
    )
    blocked = unknown.invoke("随便", {"patient_id": "p001"})
    assert blocked["plan"]["steps"] == [{"subgraph": "diabetes_guideline_qa", "parallel_group": None}]


def test_agent_and_subgraph_can_run_alone():
    app = build_app()
    agent_result = app.invoke_agent("patient_data", {"patient_id": "p001"}, user_input="补齐档案")
    assert agent_result.output["patient_profile"]["name"] == "张三"
    assert agent_result.output["medication_plan"][0]["name"] == "二甲双胍"
    assert agent_result.trace[0]["kind"] == "agent"
    assert agent_result.trace[0]["state_after"]["patient_profile"]["patient_id"] == "p001"

    subgraph_result = app.invoke_subgraph(
        "diabetes_guideline_qa",
        {"patient_id": "p001"},
        user_input="二甲双胍副作用是什么？",
    )
    assert subgraph_result["shared_context"]["compliance_passed"] is True
    assert "胃肠道" in subgraph_result["shared_context"]["guideline_answer"]
    assert subgraph_result["trace"][0]["kind"] == "agent"
