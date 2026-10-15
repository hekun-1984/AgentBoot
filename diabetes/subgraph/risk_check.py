"""先取数，再评估。主图选错时，这里仍然只走这两步。"""

from agentboot import BaseSubgraph


class DiabetesRiskCheckSubgraph(BaseSubgraph):
    name = "diabetes_risk_check"
    description = "拉取患者血糖、用药和病史，判断低血糖、高血糖和是否需要人工介入。"
    reads = ["patient_id", "user_input", "mentioned_glucose"]
    writes = [
        "patient_profile",
        "glucose_records",
        "medication_plan",
        "glucose_summary",
        "missing_fields",
        "risk_level",
        "risk_reason",
        "needs_human",
        "critical",
    ]
    steps = ["patient_data", "risk_check"]
