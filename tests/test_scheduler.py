"""调度器测试：分批、并行、失败中止和示例业务闭环。"""

from __future__ import annotations

import asyncio

import pytest

from pathlib import Path

from agentboot.app import AgentApp
from agentboot.exceptions import PlanValidationError
from agentboot.graph import BaseSubGraph
from agentboot.models import ExecutionPlan, Step
from agentboot.scheduler import hydrate_plan
from agents.diabetes_agent import DiabetesAgent
from agents.finance_agent import FinanceAgent

ROOT = Path(__file__).resolve().parents[1]


def test_batches_follow_dependencies_not_input_order(app: AgentApp) -> None:
    plan = ExecutionPlan(
        plan_id="order",
        agent="finance_agent",
        steps=[
            Step(id="report_writer", type="subgraph", depends_on=["statement_analyzer", "portfolio_risk"]),
            Step(id="portfolio_risk", type="subgraph", depends_on=["market_data_fetch"]),
            Step(id="statement_analyzer", type="subgraph", depends_on=["market_data_fetch"]),
            Step(id="market_data_fetch", type="subgraph"),
        ],
    )
    hydrate_plan(plan, app.registry)
    batches = app.scheduler.explain(plan)
    assert batches[0] == ["market_data_fetch"]
    assert set(batches[1]) == {"statement_analyzer", "portfolio_risk"}
    assert batches[2] == ["report_writer"]


def test_parallel_steps_overlap(app: AgentApp) -> None:
    started: dict[str, bool] = {}
    release = asyncio.Event()

    class Left(BaseSubGraph):
        name = "left"
        parallel_group = "g"

        async def execute(self, context: dict) -> dict:
            started["left"] = True
            if "right" not in started:
                await asyncio.wait_for(release.wait(), timeout=1)
            return {"ok": True}

    class Right(BaseSubGraph):
        name = "right"
        parallel_group = "g"

        async def execute(self, context: dict) -> dict:
            started["right"] = True
            release.set()
            return {"ok": True}

    app.registry.register_subgraph("left", Left)
    app.registry.register_subgraph("right", Right)
    plan = ExecutionPlan(
        plan_id="parallel",
        agent="demo",
        steps=[Step(id="left", type="subgraph"), Step(id="right", type="subgraph")],
    )
    result = asyncio.run(app.scheduler.execute(plan, {}))
    assert result.status == "success"
    assert result.results["left"]["ok"] is True
    assert result.results["right"]["ok"] is True


def test_failure_skips_downstream(app: AgentApp) -> None:
    class Boom(BaseSubGraph):
        name = "boom"

        async def execute(self, context: dict) -> dict:
            raise RuntimeError("boom")

    class After(BaseSubGraph):
        name = "after"
        depends_on = ["boom"]

        async def execute(self, context: dict) -> dict:
            return {"ran": True}

    app.registry.register_subgraph("boom", Boom)
    app.registry.register_subgraph("after", After)
    plan = ExecutionPlan(
        plan_id="fail",
        agent="demo",
        steps=[
            Step(id="boom", type="subgraph"),
            Step(id="after", type="subgraph", depends_on=["boom"]),
        ],
    )
    result = asyncio.run(app.scheduler.execute(plan, {}))
    assert result.status == "failed"
    assert "boom" in result.errors["boom"]
    assert "after" not in result.results
    skipped = [item for item in result.trace if item.get("step") == "after"]
    assert skipped[0]["status"] == "skipped"


def test_step_timeout(app: AgentApp) -> None:
    class Slow(BaseSubGraph):
        name = "slow"

        async def execute(self, context: dict) -> dict:
            await asyncio.sleep(1)
            return {"ok": True}

    app.registry.register_subgraph("slow", Slow)
    plan = ExecutionPlan(
        plan_id="timeout",
        agent="demo",
        steps=[Step(id="slow", type="subgraph", timeout=0.05)],
    )
    result = asyncio.run(app.scheduler.execute(plan, {}))
    assert result.status == "failed"
    assert "timed out" in result.errors["slow"]


def test_declared_skill_outside_subgraph_is_denied(app: AgentApp) -> None:
    class Local(BaseSubGraph):
        name = "local_step"
        allowed_skills = ["patient_record"]

        async def execute(self, context: dict) -> dict:
            return await self.call_skill("history_query", {"patient_id": "P001"})

    app.registry.register_subgraph("local_step", Local)
    app.router.load_table({"demo": {"allowed_skills": ["*"]}})
    plan = ExecutionPlan(
        plan_id="local",
        agent="demo",
        steps=[Step(id="local_step", type="subgraph")],
    )
    result = asyncio.run(app.scheduler.execute(plan, {}))
    assert result.status == "failed"
    assert "cannot call skill" in result.errors["local_step"]


def test_route_denies_before_execution(app: AgentApp) -> None:
    class Rogue(BaseSubGraph):
        name = "rogue_step"
        allowed_skills = ["finance_analysis"]

        async def execute(self, context: dict) -> dict:
            raise AssertionError("should be denied before execute")

    app.registry.register_subgraph("rogue_step", Rogue)
    plan = ExecutionPlan(
        plan_id="rogue",
        agent="diabetes_agent",
        steps=[Step(id="rogue_step", type="subgraph")],
    )
    result = asyncio.run(app.scheduler.execute(plan, {}))
    assert result.status == "failed"
    assert "denied" in result.errors["rogue_step"]


def test_condition_step(app: AgentApp) -> None:
    plan = ExecutionPlan(
        plan_id="cond",
        agent="demo",
        steps=[
            Step(
                id="check",
                type="condition",
                config={"path": "score", "op": "gte", "value": 3},
            )
        ],
    )
    result = asyncio.run(app.scheduler.execute(plan, {"score": 5}))
    assert result.results["check"]["passed"] is True


def test_diabetes_high_and_low_risk(app: AgentApp) -> None:
    high = asyncio.run(app.execute_plan("diabetes_agent", {"patient_id": "P001"}))
    low = asyncio.run(app.execute_plan("diabetes_agent", {"patient_id": "P002"}))
    assert high.status == "success"
    assert high.results["risk_assessor"]["risk_level"] == "high"
    assert high.results["history_analyzer"]["analysis"]["hba1c_delta"] == 0.2
    assert "不能用于临床决策" in high.results["risk_assessor"]["disclaimer"]
    batches = next(item["batches"] for item in high.trace if "batches" in item)
    assert batches[0] == ["patient_data_fetch"]
    assert set(batches[1]) == {"history_analyzer", "risk_assessor"}
    assert low.results["risk_assessor"]["risk_level"] == "low"


def test_unknown_patient_skips_downstream(app: AgentApp) -> None:
    result = asyncio.run(app.execute_plan("diabetes_agent", {"patient_id": "NOPE"}))
    assert result.status == "failed"
    assert "NOPE" in result.errors["patient_data_fetch"]
    statuses = {item["step"]: item["status"] for item in result.trace if "step" in item}
    assert statuses["history_analyzer"] == "skipped"
    assert statuses["risk_assessor"] == "skipped"


def test_finance_flow(app: AgentApp) -> None:
    high = asyncio.run(app.execute_plan("finance_agent", {"account_id": "A100"}))
    low = asyncio.run(app.execute_plan("finance_agent", {"account_id": "A200"}))
    assert high.status == "success"
    assert high.results["portfolio_risk"]["risk"]["risk_level"] == "high"
    assert high.results["statement_analyzer"]["statement"]["profit"] == 34.0
    assert "A100" in high.results["report_writer"]["report"]
    batches = next(item["batches"] for item in high.trace if "batches" in item)
    assert len(batches) == 3
    assert low.results["portfolio_risk"]["risk"]["risk_level"] == "low"
    assert "不能用于投资决策" in low.results["report_writer"]["disclaimer"]


def test_route_step_limit(app: AgentApp) -> None:
    app.router.load_table({"diabetes_agent": {"allowed_skills": ["*"], "max_steps": 1}})
    with pytest.raises(PlanValidationError, match="limit"):
        asyncio.run(app.execute_plan("diabetes_agent", {"patient_id": "P001"}))


def test_langgraph_engine_runs_parallel_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    if importlib.util.find_spec("langgraph") is None:
        pytest.skip("langgraph not installed")
    monkeypatch.setenv("AGENTBOOT_ENGINE", "langgraph")
    monkeypatch.delenv("AGENTBOOT_LLM_API_KEY", raising=False)
    application = AgentApp(app_name="lg", project_root=ROOT)
    application.register_agent(DiabetesAgent)
    application.register_agent(FinanceAgent)
    application.bootstrap()
    result = asyncio.run(application.execute_plan("finance_agent", {"account_id": "A100"}))
    assert result.status == "success"
    assert result.trace[0]["engine"] == "langgraph"
    assert result.results["statement_analyzer"]["statement"]["profit"] == 34.0
    assert result.results["portfolio_risk"]["risk"]["risk_level"] == "high"
    assert "A100" in result.results["report_writer"]["report"]
