"""费用审批主图。模型只选子图，超限由声明的字段兜底。"""

from agentboot import AgentBootApplication, BaseGraph


class ExpenseApprovalGraph(BaseGraph):
    name = "expense_approval"
    description = "费用审批编排。主图只选择子图，不直接执行 Tool。"
    subgraphs = ["expense_review", "expense_escalation", "policy_reference"]
    fallback_subgraph = "policy_reference"
    max_steps = 4
    guard_field = "decision"
    guard_values = ("escalate", "reject")
    guard_subgraph = "expense_escalation"

    def offline_route(self, user_input: str, catalog: list[dict], shared_context: dict) -> dict:
        text = user_input or ""
        if any(word in text for word in ("限额", "制度", "标准")):
            return {"steps": [{"subgraph": "policy_reference"}], "reason": "问题落在制度查询"}
        if shared_context.get("expense_id") or any(word in text for word in ("报销", "费用", "单据")):
            return {"steps": [{"subgraph": "expense_review"}], "reason": "先审这一张单据"}
        return {"steps": [], "reason": "没有匹配到明确能力"}

    def screen(self, user_input: str, shared_context: dict) -> dict:
        return {"user_input": user_input, "force_emergency": False}

    def render_answer(self, state) -> str:
        context = state.get("shared_context") or {}
        if context.get("notice"):
            return str(context["notice"])
        if context.get("decision"):
            return f"审批结论：{context['decision']}。{context.get('policy_reason') or ''}"
        if context.get("policy_text"):
            return str(context["policy_text"])
        return "没有形成可用回复。"


def build_app(model=None, router=None, trace_sink=None, models=None) -> AgentBootApplication:
    """装配费用审批示例。不传模型时，Agent 与路由都走可重复的离线实现。"""
    app = AgentBootApplication(model=model, router=router, trace_sink=trace_sink, models=models)
    return app.scan("expense")
