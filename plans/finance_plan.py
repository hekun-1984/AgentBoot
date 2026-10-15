"""财务分析执行计划。"""

from __future__ import annotations

from typing import Any

from agentboot.models import ExecutionPlan, Step
from plans.base_plan import BasePlan


class FinancePlan(BasePlan):
    agent = "finance_agent"

    def build(self, context: dict[str, Any]) -> ExecutionPlan:
        account_id = context.get("account_id", "unknown")
        return ExecutionPlan(
            plan_id=f"finance_{account_id}",
            agent=self.agent,
            steps=[
                Step(id="market_data_fetch", type="subgraph"),
                Step(
                    id="statement_analyzer",
                    type="subgraph",
                    depends_on=["market_data_fetch"],
                ),
                Step(
                    id="portfolio_risk",
                    type="subgraph",
                    depends_on=["market_data_fetch"],
                ),
                Step(
                    id="report_writer",
                    type="subgraph",
                    depends_on=["statement_analyzer", "portfolio_risk"],
                ),
            ],
            metadata={"source": "template", "template": "FinancePlan"},
        )
