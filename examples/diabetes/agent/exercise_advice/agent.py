"""运动建议前先过危急值规则。"""

from pathlib import Path

from agentboot import BaseAgent
from diabetes.agent.common import calls, data_of, final, payloads


def _value_text(shared_context: dict) -> str:
    summary = shared_context.get("glucose_summary") or {}
    value = shared_context.get("mentioned_glucose")
    if value is None:
        value = summary.get("mentioned_value")
    if value is None:
        value = summary.get("latest_value")
    return "" if value is None else str(value)


class ExerciseAdviceAgent(BaseAgent):
    name = "exercise_advice"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        text = str(shared_context.get("user_input") or "")
        glucose_text = _value_text(shared_context)
        fasting = bool((shared_context.get("intake") or {}).get("fasting_mentioned") or "空腹" in text)
        if glucose_text and "check_critical_value" not in found:
            return calls(
                (
                    "check_critical_value",
                    {"glucose_value": float(glucose_text), "symptoms": text, "fasting": fasting},
                    "critical",
                )
            )
        assessment = data_of(found, "check_critical_value")
        level = assessment.get("level") or str(shared_context.get("risk_level") or "low")
        if "generate_exercise_suggestion" not in found:
            preference = ((shared_context.get("patient_profile") or {}).get("preferences") or {}).get("exercise") or ""
            return calls(
                (
                    "generate_exercise_suggestion",
                    {"risk_level": level, "preference": preference, "glucose_text": glucose_text},
                    "exercise",
                )
            )
        suggestion = data_of(found, "generate_exercise_suggestion")
        return final({"exercise_suggestion": suggestion.get("text") or ""})
