"""白名单路由配置。开发者在此定义每个 Agent 可访问的能力白名单。"""

ROUTE_TABLE = {
    "diabetes_agent": {
        "allowed_skills": [
            "medical_analysis",
            "history_query",
            "risk_model",
            "notification",
            "patient_record",
        ],
        "max_steps": 10,
        "timeout": 120,
    },
    "finance_agent": {
        "allowed_skills": [
            "finance_analysis",
            "risk_assessment",
            "report_generate",
            "data_query",
        ],
        "max_steps": 8,
        "timeout": 90,
    },
}
