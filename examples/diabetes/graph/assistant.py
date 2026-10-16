"""糖尿病管理助手主图。模型只选子图，危急值由规则提前截断。"""

from agentboot import AgentBootApplication, BaseGraph

from diabetes.graph.glucose import extract_glucose


class DiabetesAssistantGraph(BaseGraph):
    name = "diabetes_assistant"
    description = "糖尿病管理编排。主图只选择子图，不直接执行 Tool。"
    subgraphs = [
        "diabetes_risk_check",
        "diabetes_guideline_qa",
        "diabetes_management_plan",
        "emergency_escalation",
    ]
    fallback_subgraph = "diabetes_guideline_qa"
    emergency_subgraph = "emergency_escalation"
    max_steps = 6
    high_risk_levels = ("high", "critical")
    guard_field = "risk_level"
    guard_values = ("high", "critical")
    guard_subgraph = "emergency_escalation"
    emergency_symptoms = ("昏迷", "意识不清", "胸痛", "抽搐", "叫不醒", "手抖")
    critical_tool = "check_critical_value"

    def offline_route(self, user_input: str, catalog: list[dict], shared_context: dict) -> dict:
        text = user_input or ""
        lowered = text.lower()
        if "指南" in text or "副作用" in text or "hba1c" in lowered or "糖化" in text:
            return {
                "steps": [{"subgraph": "diabetes_guideline_qa"}],
                "reason": "问题落在指南问答",
            }
        if any(word in text for word in ("饮食", "运动", "复查")):
            return {
                "steps": [
                    {"subgraph": "diabetes_risk_check", "parallel_group": "care"},
                    {"subgraph": "diabetes_management_plan", "parallel_group": "care"},
                ],
                "reason": "风险评估与管理计划并行",
            }
        if "血糖" in text:
            return {
                "steps": [{"subgraph": "diabetes_risk_check"}],
                "reason": "先做风险评估",
            }
        return {"steps": [], "reason": "没有匹配到明确能力"}

    def screen(self, user_input: str, shared_context: dict) -> dict:
        value = extract_glucose(user_input)
        force = any(word in user_input for word in self.emergency_symptoms)
        assessment = None
        tool = self._tools.get(self.critical_tool) if self.critical_tool else None
        if value is not None and tool is not None:
            assessment = tool.fn(
                glucose_value=value,
                symptoms=user_input,
                fasting="空腹" in user_input,
            )
            if assessment.get("level") in self.high_risk_levels:
                force = True
        context = {
            "user_input": user_input,
            "force_emergency": force,
            "intake": {
                "mentioned_glucose": value,
                "fasting_mentioned": "空腹" in user_input,
                "assessment": assessment,
                "force_emergency": force,
            },
        }
        if value is not None:
            context["mentioned_glucose"] = value
        return context

    def render_answer(self, state) -> str:
        context = state.get("shared_context") or {}
        results = state.get("results") or {}
        if "emergency_escalation" in results:
            return (
                context.get("emergency_message")
                or (results.get("emergency_escalation") or {}).get("emergency_message")
                or "请立即就医，并已转人工。"
            )
        parts: list[str] = []
        if context.get("risk_level"):
            parts.append(f"风险评估：{context['risk_level']}。{context.get('risk_reason') or ''}")
        if context.get("risk_level") in self.high_risk_levels:
            parts.append("风险较高，请尽快就医。本次不提供饮食、运动或用药调整。")
            return "\n".join(parts)
        plan = context.get("management_plan") or {}
        if plan.get("diet"):
            parts.append(f"饮食：{plan['diet']}")
        if plan.get("exercise"):
            parts.append(f"运动：{plan['exercise']}")
        follow = plan.get("followup") or {}
        if follow.get("message"):
            parts.append(f"复查：{follow.get('remind_at') or ''} {follow['message']}".strip())
        if "diabetes_guideline_qa" in results and context.get("guideline_answer"):
            parts.append(str(context["guideline_answer"]))
            titles = "；".join(
                item.get("title", "")
                for item in context.get("sources") or []
                if isinstance(item, dict) and item.get("title")
            )
            if titles:
                parts.append(f"来源：{titles}")
        if not parts:
            parts.append("没有形成可用回复。")
        return "\n".join(parts)


def build_app(model=None, router=None, trace_sink=None, models=None) -> AgentBootApplication:
    """装配糖尿病示例。不传模型时，Agent 与路由都走可重复的离线实现。"""
    app = AgentBootApplication(model=model, router=router, trace_sink=trace_sink, models=models)
    return app.scan("diabetes")
