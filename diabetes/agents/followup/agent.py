"""复查提醒。高危才记风险事件。"""

from datetime import date, timedelta
from pathlib import Path

from agentboot import BaseAgent
from diabetes.agents.common import calls, data_of, final, payloads


def _days(shared_context: dict) -> int:
    level = shared_context.get("risk_level")
    if level in {"high", "critical"}:
        return 1
    summary = shared_context.get("glucose_summary") or {}
    value = summary.get("mentioned_value")
    if value is None:
        value = summary.get("latest_fasting")
    if value is None:
        value = summary.get("latest_value")
    text = str(shared_context.get("user_input") or "")
    fasting = bool((shared_context.get("intake") or {}).get("fasting_mentioned") or "空腹" in text)
    if fasting or (value is not None and float(value) >= 7.0):
        return 14
    return 90


class FollowupAgent(BaseAgent):
    name = "followup"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        patient_id = str(shared_context.get("patient_id") or "")
        if "send_followup_reminder" not in found:
            remind_at = (date.today() + timedelta(days=_days(shared_context))).isoformat()
            planned = [
                ("get_medication_plan", {"patient_id": patient_id}, "meds"),
                (
                    "send_followup_reminder",
                    {
                        "patient_id": patient_id,
                        "remind_at": remind_at,
                        "message": "请复查空腹血糖。不要自行调整药物剂量。",
                    },
                    "remind",
                ),
            ]
            if shared_context.get("risk_level") in {"high", "critical"}:
                planned.append(
                    (
                        "log_risk_event",
                        {
                            "patient_id": patient_id,
                            "risk_type": str(shared_context.get("risk_level")),
                            "description": str(shared_context.get("risk_reason") or "高危随访"),
                        },
                        "event",
                    )
                )
            return calls(*planned)
        reminder = data_of(found, "send_followup_reminder")
        medication = data_of(found, "get_medication_plan")
        event = data_of(found, "log_risk_event")
        return final(
            {
                "followup_plan": {
                    "patient_id": patient_id,
                    "medication": medication.get("medications") or [],
                    "remind_at": reminder.get("remind_at"),
                    "message": reminder.get("message"),
                    "sent": bool(reminder.get("sent")),
                    "risk_event_id": event.get("record_id"),
                }
            }
        )
