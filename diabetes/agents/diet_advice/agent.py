"""饮食建议只包装 Tool 的结果。"""

from pathlib import Path

from agentboot import BaseAgent
from diabetes.agents.common import calls, data_of, final, payloads


def _glucose_text(shared_context: dict) -> str:
    summary = shared_context.get("glucose_summary") or {}
    value = summary.get("mentioned_value")
    if value is None:
        value = summary.get("latest_fasting")
    if value is None:
        value = summary.get("latest_value")
    if value is None:
        value = shared_context.get("mentioned_glucose")
    return "" if value is None else str(value)


class DietAdviceAgent(BaseAgent):
    name = "diet_advice"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        if "generate_diet_suggestion" not in found:
            profile = shared_context.get("patient_profile") or {}
            preference = (profile.get("preferences") or {}).get("diet") or ""
            text = str(shared_context.get("user_input") or "")
            fasting = bool((shared_context.get("intake") or {}).get("fasting_mentioned") or "空腹" in text)
            return calls(
                (
                    "generate_diet_suggestion",
                    {
                        "risk_level": str(shared_context.get("risk_level") or "low"),
                        "preference": preference,
                        "glucose_text": _glucose_text(shared_context),
                        "fasting": fasting,
                    },
                    "diet",
                ),
                ("query_guideline_rag", {"question": text or "糖尿病饮食"}, "rag"),
            )
        suggestion = data_of(found, "generate_diet_suggestion")
        chunks = list(data_of(found, "query_guideline_rag").get("chunks") or [])
        return final({"diet_suggestion": suggestion.get("text") or "", "diet_sources": chunks})
