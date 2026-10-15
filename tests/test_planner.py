"""计划生成和依赖校验测试。"""

from __future__ import annotations

import asyncio
import json

import pytest

from agentboot.app import AgentApp
from agentboot.exceptions import PlanValidationError
from agentboot.models import ExecutionPlan, Step
from agentboot.planner import parse_plan_payload
from agentboot.validator import PlanValidator


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text


class _BrokenLLM:
    async def complete(self, prompt: str) -> str:
        raise RuntimeError("model down")


def _validator(app: AgentApp) -> PlanValidator:
    return PlanValidator(set(app.registry.list_subgraphs()), set(app.registry.list_skills()))


def test_parse_fenced_plan() -> None:
    payload = parse_plan_payload(
        '说明如下：\n```json\n{"plan_id": "p", "steps": [{"id": "a"}]}\n```'
    )
    assert payload["plan_id"] == "p"
    assert payload["steps"][0]["id"] == "a"


def test_parallel_groups_follow_declaration() -> None:
    plan = ExecutionPlan(
        plan_id="g",
        agent="demo",
        steps=[
            Step(id="a", type="subgraph", parallel_group="collect"),
            Step(id="b", type="subgraph"),
        ],
    )
    groups = plan.get_parallel_groups()
    assert [step.id for step in groups["collect"]] == ["a"]
    assert [step.id for step in groups["default"]] == ["b"]


def test_validator_rejects_cycle_duplicate_and_missing(app: AgentApp) -> None:
    validator = _validator(app)
    with pytest.raises(PlanValidationError, match="Duplicate"):
        validator.validate(
            ExecutionPlan(
                plan_id="dup",
                agent="diabetes_agent",
                steps=[
                    Step(id="patient_data_fetch", type="subgraph"),
                    Step(id="patient_data_fetch", type="subgraph"),
                ],
            )
        )
    with pytest.raises(PlanValidationError, match="unknown step"):
        validator.validate(
            ExecutionPlan(
                plan_id="missing",
                agent="diabetes_agent",
                steps=[Step(id="patient_data_fetch", type="subgraph", depends_on=["missing"])],
            )
        )
    with pytest.raises(PlanValidationError, match="Circular"):
        validator.validate(
            ExecutionPlan(
                plan_id="cycle",
                agent="diabetes_agent",
                steps=[
                    Step(id="patient_data_fetch", type="subgraph", depends_on=["history_analyzer"]),
                    Step(id="history_analyzer", type="subgraph", depends_on=["patient_data_fetch"]),
                ],
            )
        )


def test_validator_rejects_parallel_group_conflict(app: AgentApp) -> None:
    validator = _validator(app)
    with pytest.raises(PlanValidationError, match="conflicting"):
        validator.validate(
            ExecutionPlan(
                plan_id="conflict",
                agent="diabetes_agent",
                steps=[
                    Step(id="patient_data_fetch", type="subgraph", parallel_group="analysis_phase"),
                    Step(
                        id="history_analyzer",
                        type="subgraph",
                        parallel_group="analysis_phase",
                        depends_on=["patient_data_fetch"],
                    ),
                ],
            )
        )


def test_validator_rejects_unregistered_subgraph(app: AgentApp) -> None:
    with pytest.raises(PlanValidationError, match="not registered"):
        _validator(app).validate(
            ExecutionPlan(
                plan_id="ghost",
                agent="diabetes_agent",
                steps=[Step(id="does_not_exist", type="subgraph")],
            )
        )


def test_unreachable_component_is_reported() -> None:
    validator = PlanValidator(set())
    with pytest.raises(PlanValidationError, match="Unreachable"):
        validator._check_unreachable(
            [
                Step(id="a", type="llm", depends_on=["b"]),
                Step(id="b", type="llm", depends_on=["a"]),
            ]
        )


def test_template_discovery(app: AgentApp) -> None:
    assert set(app.planner.templates) == {"diabetes_agent", "finance_agent"}


def test_llm_plan_is_used_when_intent_exists(app: AgentApp) -> None:
    client = _FakeLLM(
        json.dumps(
            {
                "plan_id": "llm_demo",
                "steps": [
                    {"id": "patient_data_fetch", "type": "subgraph"},
                    {
                        "id": "risk_assessor",
                        "type": "subgraph",
                        "depends_on": ["patient_data_fetch"],
                    },
                ],
            }
        )
    )
    app.planner.llm_client = client
    result = asyncio.run(
        app.execute_plan("diabetes_agent", {"intent": "只评估风险", "patient_id": "P001"})
    )
    assert result.status == "success"
    assert result.plan_id == "llm_demo"
    assert "history_analyzer" not in result.results
    assert result.results["risk_assessor"]["risk_level"] == "high"
    assert "Allowed steps" in client.prompts[0]
    assert next(item["source"] for item in result.trace if item.get("event") == "plan") == "llm"


def test_llm_failure_falls_back_to_template(app: AgentApp) -> None:
    app.planner.llm_client = _BrokenLLM()
    result = asyncio.run(
        app.execute_plan("diabetes_agent", {"intent": "评估", "patient_id": "P002"})
    )
    assert result.status == "success"
    assert result.plan_id == "diabetes_P002"
    assert "history_analyzer" in result.results
    assert next(item["source"] for item in result.trace if item.get("event") == "plan") == "template"
