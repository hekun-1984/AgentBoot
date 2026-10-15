"""糖尿病管理执行计划。"""

from __future__ import annotations

from typing import Any

from agentboot.models import ExecutionPlan, Step
from plans.base_plan import BasePlan


class DiabetesPlan(BasePlan):
    agent = "diabetes_agent"

    def build(self, context: dict[str, Any]) -> ExecutionPlan:
        patient_id = context.get("patient_id", "unknown")
        return ExecutionPlan(
            plan_id=f"diabetes_{patient_id}",
            agent=self.agent,
            steps=[
                Step(id="patient_data_fetch", type="subgraph"),
                Step(
                    id="history_analyzer",
                    type="subgraph",
                    depends_on=["patient_data_fetch"],
                ),
                Step(
                    id="risk_assessor",
                    type="subgraph",
                    depends_on=["patient_data_fetch"],
                ),
            ],
            metadata={"source": "template", "template": "DiabetesPlan"},
        )
