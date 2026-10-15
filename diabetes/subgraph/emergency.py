"""只做紧急提示和人工升级，不生成用药调整。"""

from langgraph.graph import END, START

from agentboot import BaseSubgraph


class EmergencyEscalationSubgraph(BaseSubgraph):
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
