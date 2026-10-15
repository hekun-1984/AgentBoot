"""饮食、运动、复查按固定顺序执行，最后汇总。不在这里做自由路由。"""

from langgraph.graph import END

from agentboot import BaseSubgraph


class DiabetesManagementPlanSubgraph(BaseSubgraph):
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
