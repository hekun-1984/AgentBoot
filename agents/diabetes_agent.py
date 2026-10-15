"""示例：糖尿病管理 Agent。展示如何注册子图并交出一份执行计划。"""

from __future__ import annotations

from typing import Any

from agentboot.graph import BaseSubGraph
from agentboot.models import ExecutionPlan
from agents.base_agent import BaseAgent
from plans.diabetes_plan import DiabetesPlan
from skills.medical_skill import (
    HistoryQuery,
    MedicalAnalysis,
    Notification,
    PatientRecord,
    RiskModel,
)


class PatientDataFetcher(BaseSubGraph):
    """子图：获取患者数据。无依赖，可作为并行采集的起点。"""

    name = "patient_data_fetch"
    description = "读取患者档案和实验室结果"
    parallel_group = "data_collection"
    allowed_skills = ["patient_record"]

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        record = await self.call_skill(
            "patient_record",
            {"patient_id": context.get("patient_id"), "labs": context.get("labs")},
        )
        labs = record["labs"]
        return {
            "patient_data": record,
            "lab_results": [
                {"name": "hba1c", "value": labs.get("hba1c")},
                {"name": "fasting_glucose", "value": labs.get("fasting_glucose")},
                {"name": "bmi", "value": labs.get("bmi")},
            ],
        }


class HistoryAnalyzer(BaseSubGraph):
    """子图：分析历史病历。依赖患者数据采集。"""

    name = "history_analyzer"
    description = "结合历史病历分析糖化血红蛋白趋势"
    depends_on = ["patient_data_fetch"]
    parallel_group = "analysis_phase"
    allowed_skills = ["medical_analysis", "history_query"]

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        patient = context["patient_data_fetch"]["patient_data"]
        history = await self.call_skill("history_query", {"patient_id": patient["patient_id"]})
        analysis = await self.call_skill(
            "medical_analysis",
            {"labs": patient["labs"], "records": history["records"]},
        )
        return {"analysis": analysis}


class RiskAssessor(BaseSubGraph):
    """子图：风险评估。与病历分析同组并行，只依赖患者数据。"""

    name = "risk_assessor"
    description = "评估糖尿病风险并生成提醒"
    depends_on = ["patient_data_fetch"]
    parallel_group = "analysis_phase"
    allowed_skills = ["risk_model", "notification"]

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        patient = context["patient_data_fetch"]["patient_data"]
        risk = await self.call_skill("risk_model", {"labs": patient["labs"]})
        note = await self.call_skill(
            "notification",
            {"patient_id": patient["patient_id"], "risk_level": risk["risk_level"]},
        )
        return {
            "risk_level": risk["risk_level"],
            "score": risk["score"],
            "recommendations": risk["recommendations"],
            "notification": note["message"],
            "disclaimer": risk["disclaimer"],
        }


class DiabetesAgent(BaseAgent):
    """糖尿病管理 Agent。开发者继承 BaseAgent，注册子图，并提供执行计划。"""

    name = "diabetes_agent"
    description = "糖尿病管理助手：数据采集 -> 病历分析 -> 风险评估"

    def __init__(self) -> None:
        super().__init__()
        self.register_skill(PatientRecord)
        self.register_skill(HistoryQuery)
        self.register_skill(MedicalAnalysis)
        self.register_skill(RiskModel)
        self.register_skill(Notification)
        self.register_subgraph(PatientDataFetcher)
        self.register_subgraph(HistoryAnalyzer)
        self.register_subgraph(RiskAssessor)

    def sample_context(self) -> dict[str, Any]:
        return {"patient_id": "P001"}

    async def execute_plan(self, context: dict[str, Any]) -> ExecutionPlan:
        return DiabetesPlan().build(context)
