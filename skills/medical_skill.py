"""示例：医疗领域可复用能力。数据是虚构的，不能用于临床决策。"""

from __future__ import annotations

from typing import Any

from skills.base_skill import BaseSkill

DISCLAIMER = "演示数据，不能用于临床决策。"

PATIENTS: dict[str, dict[str, Any]] = {
    "P001": {
        "patient_id": "P001",
        "name": "演示患者甲",
        "age": 58,
        "labs": {"hba1c": 8.6, "fasting_glucose": 11.2, "bmi": 29.1},
        "history": [
            {"date": "2025-11-02", "hba1c": 8.1, "note": "血糖控制不佳"},
            {"date": "2026-03-14", "hba1c": 8.4, "note": "药物依从性一般"},
        ],
    },
    "P002": {
        "patient_id": "P002",
        "name": "演示患者乙",
        "age": 46,
        "labs": {"hba1c": 6.1, "fasting_glucose": 5.6, "bmi": 22.4},
        "history": [
            {"date": "2026-01-09", "hba1c": 6.3, "note": "生活方式干预后稳定"},
        ],
    },
}


def _patient(context: dict[str, Any]) -> dict[str, Any]:
    patient_id = str(context.get("patient_id") or "")
    record = PATIENTS.get(patient_id)
    override = context.get("labs")
    if record is None:
        if not isinstance(override, dict):
            raise ValueError(f"未知患者: {patient_id}")
        record = {
            "patient_id": patient_id,
            "name": "外部患者",
            "age": context.get("age"),
            "labs": dict(override),
            "history": list(context.get("history") or []),
        }
    elif isinstance(override, dict):
        record = {**record, "labs": {**record["labs"], **override}}
    return record


def assess_risk(labs: dict[str, Any]) -> dict[str, Any]:
    hba1c = float(labs.get("hba1c") or 0)
    glucose = float(labs.get("fasting_glucose") or 0)
    bmi = float(labs.get("bmi") or 0)
    score = 0
    if hba1c >= 8:
        score += 2
    elif hba1c >= 6.5:
        score += 1
    if glucose >= 10:
        score += 2
    elif glucose >= 7:
        score += 1
    if bmi >= 28:
        score += 1
    if score >= 3:
        level = "high"
        recommendations = ["建议尽快复诊，复查糖化血红蛋白，并评估用药方案。"]
    elif score >= 1:
        level = "medium"
        recommendations = ["建议加强饮食与运动管理，4 到 8 周后复查。"]
    else:
        level = "low"
        recommendations = ["建议维持当前方案，按期监测血糖。"]
    return {
        "risk_level": level,
        "score": score,
        "recommendations": recommendations,
        "disclaimer": DISCLAIMER,
    }


class PatientRecord(BaseSkill):
    """患者档案能力。"""

    name = "patient_record"
    description = "读取患者基本信息和最近一次实验室结果"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        record = _patient(context)
        return {
            "patient_id": record["patient_id"],
            "name": record["name"],
            "age": record["age"],
            "labs": dict(record["labs"]),
            "disclaimer": DISCLAIMER,
        }


class HistoryQuery(BaseSkill):
    """病历查询能力。"""

    name = "history_query"
    description = "查询患者历史病历记录"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        record = _patient(context)
        return {"records": list(record["history"]), "disclaimer": DISCLAIMER}


class MedicalAnalysis(BaseSkill):
    """医疗分析能力。"""

    name = "medical_analysis"
    description = "对医疗数据进行智能分析"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        labs = dict(context.get("labs") or {})
        records = list(context.get("records") or [])
        current = float(labs.get("hba1c") or 0)
        previous = [float(item["hba1c"]) for item in records if "hba1c" in item]
        delta = round(current - previous[-1], 2) if previous else 0.0
        flags: list[str] = []
        if current >= 8:
            flags.append("hba1c_high")
        if delta > 0:
            flags.append("hba1c_rising")
        trend = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
        summary = f"最近一次糖化血红蛋白 {current}%，较上一条记录{trend} {abs(delta)} 个百分点。"
        return {
            "summary": summary,
            "hba1c_delta": delta,
            "flags": flags,
            "disclaimer": DISCLAIMER,
        }


class RiskModel(BaseSkill):
    """糖尿病风险模型。"""

    name = "risk_model"
    description = "根据糖化血红蛋白、空腹血糖和 BMI 评估风险"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        return assess_risk(dict(context.get("labs") or {}))


class Notification(BaseSkill):
    """通知文案能力。"""

    name = "notification"
    description = "把风险评估整理成可阅读的提醒"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        patient_id = context.get("patient_id", "")
        level = context.get("risk_level", "unknown")
        return {
            "message": f"患者 {patient_id} 风险评估为 {level}，请查阅建议。",
            "disclaimer": DISCLAIMER,
        }
