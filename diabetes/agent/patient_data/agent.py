"""采集患者档案、血糖和用药。"""

from pathlib import Path

from agentboot import BaseAgent
from diabetes.agent.common import calls, data_of, final, payloads


class PatientDataAgent(BaseAgent):
    name = "patient_data"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        patient_id = str(shared_context.get("patient_id") or "")
        if "get_patient_profile" not in found:
            return calls(
                ("get_patient_profile", {"patient_id": patient_id}, "profile"),
                ("get_glucose_records", {"patient_id": patient_id}, "glucose"),
                ("get_medication_plan", {"patient_id": patient_id}, "meds"),
            )
        profile = data_of(found, "get_patient_profile")
        glucose = data_of(found, "get_glucose_records")
        medication = data_of(found, "get_medication_plan")
        records = list(glucose.get("records") or [])
        medications = list(medication.get("medications") or [])
        missing = []
        if not profile.get("found"):
            missing.append("patient_profile")
        if not records:
            missing.append("glucose_records")
        if not medications:
            missing.append("medication_plan")
        fasting = [item["value"] for item in records if item.get("period") == "fasting"]
        latest = records[-1] if records else None
        summary = {
            "latest_value": None if latest is None else latest["value"],
            "latest_period": None if latest is None else latest.get("period"),
            "latest_fasting": fasting[-1] if fasting else None,
            "average_fasting": round(sum(fasting) / len(fasting), 2) if fasting else None,
            "mentioned_value": shared_context.get("mentioned_glucose"),
            "unit": "mmol/L",
        }
        return final(
            {
                "patient_profile": profile,
                "glucose_records": records,
                "medication_plan": medications,
                "glucose_summary": summary,
                "missing_fields": missing,
            }
        )
