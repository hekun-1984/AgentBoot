"""用规则 Tool 判断风险等级。"""

from pathlib import Path

from agentboot import BaseAgent
from diabetes.agents.common import calls, data_of, final, payloads

_EMERGENCY_SYMPTOMS = ("昏迷", "意识不清", "胸痛", "抽搐", "叫不醒", "手抖")


def _glucose_value(shared_context: dict) -> float | None:
    if shared_context.get("mentioned_glucose") is not None:
        return float(shared_context["mentioned_glucose"])
    summary = shared_context.get("glucose_summary") or {}
    if summary.get("mentioned_value") is not None:
        return float(summary["mentioned_value"])
    if summary.get("latest_value") is not None:
        return float(summary["latest_value"])
    records = shared_context.get("glucose_records") or []
    if records:
        return float(records[-1]["value"])
    return None


class RiskCheckAgent(BaseAgent):
    name = "risk_check"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        text = str(shared_context.get("user_input") or "")
        value = _glucose_value(shared_context)
        fasting = bool((shared_context.get("intake") or {}).get("fasting_mentioned") or "空腹" in text)
        if "check_critical_value" not in found and value is not None:
            return calls(
                (
                    "check_critical_value",
                    {"glucose_value": value, "symptoms": text, "fasting": fasting},
                    "critical",
                ),
                ("query_guideline_rag", {"question": text or "糖尿病风险"}, "rag"),
            )
        if "query_guideline_rag" not in found:
            return calls(("query_guideline_rag", {"question": text or "糖尿病风险"}, "rag"))
        assessment = data_of(found, "check_critical_value")
        if not assessment and any(word in text for word in _EMERGENCY_SYMPTOMS):
            assessment = {"level": "critical", "critical": True, "message": "用户描述了危急症状"}
        if not assessment:
            assessment = {"level": "low", "critical": False, "message": "没有可用的血糖读数"}
        level = assessment.get("level") or "low"
        return final(
            {
                "risk_level": level,
                "risk_reason": assessment.get("message") or "",
                "needs_human": level in {"high", "critical"},
                "critical": bool(assessment.get("critical")),
            }
        )
