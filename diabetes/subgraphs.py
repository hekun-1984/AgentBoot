"""四张固定流程的子图。边在代码里写死。"""

from langgraph.graph import END, START

from agentboot import BaseSubgraph


class DiabetesRiskCheckSubgraph(BaseSubgraph):
    """先取数，再评估。主图选错时，这里仍然只走这两步。"""

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


class DiabetesGuidelineQASubgraph(BaseSubgraph):
    """回答之后必须过合规校验，没有来源或不在范围内都不能直接返回。"""

    name = "diabetes_guideline_qa"
    description = "依据糖尿病指南回答问题。超出范围时拒绝，回答必须附带来源。"
    reads = ["user_input"]
    writes = ["guideline_answer", "sources", "in_scope", "compliance_passed"]

    def build(self, builder, runtime):
        runtime.add_agent(builder, "guideline_qa")
        runtime.add_tool(
            builder,
            "compliance",
            "check_guideline_compliance",
            read={"answer": "guideline_answer", "in_scope": "in_scope", "sources_json": "sources"},
            write={
                "guideline_answer": "answer",
                "compliance_passed": "passed",
                "sources": "sources",
            },
            json_args=("sources_json",),
        )
        builder.add_edge(START, "guideline_qa")
        builder.add_edge("guideline_qa", "compliance")
        builder.add_edge("compliance", END)


class DiabetesManagementPlanSubgraph(BaseSubgraph):
    """饮食、运动、复查按固定顺序执行，最后汇总。不在这里做自由路由。"""

    name = "diabetes_management_plan"
    description = "按固定顺序生成饮食建议、运动建议和复查提醒，并汇总成管理计划。"
    reads = ["patient_id", "user_input", "mentioned_glucose"]
    writes = [
        "patient_profile",
        "glucose_records",
        "medication_plan",
        "glucose_summary",
        "missing_fields",
        "diet_suggestion",
        "diet_sources",
        "exercise_suggestion",
        "followup_plan",
        "management_plan",
    ]

    def build(self, builder, runtime):
        last = runtime.add_sequence(builder, ["patient_data", "diet_advice", "exercise_advice", "followup"])
        builder.add_node("summarize", self.summarize)
        builder.add_edge(last, "summarize")
        builder.add_edge("summarize", END)

    def summarize(self, state):
        context = dict(state.get("shared_context") or {})
        plan = {
            "diet": context.get("diet_suggestion"),
            "exercise": context.get("exercise_suggestion"),
            "followup": context.get("followup_plan"),
        }
        context["management_plan"] = plan
        output = dict(state.get("output") or {})
        output["management_plan"] = plan
        return {"shared_context": context, "output": output}


class EmergencyEscalationSubgraph(BaseSubgraph):
    """只做紧急提示和人工升级，不生成用药调整。"""

    name = "emergency_escalation"
    description = "血糖危急或出现胸痛、昏迷、严重低血糖时记录事件并转人工。不调整药物。"
    reads = ["patient_id", "user_input", "mentioned_glucose"]
    writes = [
        "risk_level",
        "risk_reason",
        "needs_human",
        "critical",
        "risk_event_id",
        "emergency_message",
    ]

    def build(self, builder, runtime):
        runtime.add_agent(builder, "risk_check")
        runtime.add_tool(
            builder,
            "log_event",
            "log_risk_event",
            read={
                "patient_id": "patient_id",
                "risk_type": "risk_level",
                "description": "risk_reason",
            },
            write={"risk_event_id": "record_id"},
        )
        builder.add_node("compose", self.compose)
        builder.add_edge(START, "risk_check")
        builder.add_edge("risk_check", "log_event")
        builder.add_edge("log_event", "compose")
        builder.add_edge("compose", END)

    def compose(self, state):
        context = dict(state.get("shared_context") or {})
        reason = context.get("risk_reason") or "疑似急症"
        message = (
            f"{reason}。请立即停止活动。如果人还清醒，可以补充约 15 克快速升糖的碳水化合物，"
            "并马上联系急救或身边的人。不要自行加减胰岛素或口服药。事件已记录，并转人工处理。"
        )
        context["emergency_message"] = message
        context["needs_human"] = True
        output = dict(state.get("output") or {})
        output.update(
            {
                "emergency_message": message,
                "needs_human": True,
                "risk_level": context.get("risk_level"),
                "risk_event_id": context.get("risk_event_id"),
            }
        )
        return {"shared_context": context, "output": output}
