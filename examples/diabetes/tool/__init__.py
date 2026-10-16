"""Tool 层。扫描这个包即可注册全部原子能力。"""

from diabetes.tool.definitions import (
    check_critical_value,
    check_guideline_compliance,
    generate_diet_suggestion,
    generate_exercise_suggestion,
    get_glucose_records,
    get_medication_plan,
    get_patient_profile,
    log_risk_event,
    query_guideline_rag,
    send_followup_reminder,
)

__all__ = [
    "check_critical_value",
    "check_guideline_compliance",
    "generate_diet_suggestion",
    "generate_exercise_suggestion",
    "get_glucose_records",
    "get_medication_plan",
    "get_patient_profile",
    "log_risk_event",
    "query_guideline_rag",
    "send_followup_reminder",
]
