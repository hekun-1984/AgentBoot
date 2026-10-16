"""费用审批的原子能力。限额不交给模型。"""

from __future__ import annotations

from agentboot import tool
from agentboot.catalog import current_catalog
from agentboot.exceptions import ConfigurationError

from expense.tool.store import STORE

POLICY_CLAUSES = (
    "办公费两万元以内可直接通过。",
    "招待费超过五千元，或其他费用超过两万元，必须升级审批。",
    "超过十万元不予通过。",
)


@tool(name="get_expense", description="按报销单号读取金额、类别和事由")
def get_expense(expense_id: str) -> dict:
    claim = STORE.claims.get(expense_id)
    if claim is None:
        return {"found": False, "expense_id": expense_id}
    return {"found": True, **claim}


@tool(name="check_amount_policy", description="按制度判断金额是通过、升级还是拒绝")
def check_amount_policy(amount: float, category: str) -> dict:
    if amount > 100000:
        return {"decision": "reject", "policy_reason": "超过十万元上限，不予通过"}
    if category == "招待" and amount > 5000:
        return {"decision": "escalate", "policy_reason": "招待费超过五千元，需要升级审批"}
    if amount > 20000:
        return {"decision": "escalate", "policy_reason": "金额超过两万元，需要升级审批"}
    return {"decision": "pass", "policy_reason": "金额在制度限额内"}


@tool(name="record_escalation", description="把超限单据登记为升级单，不直接入账")
def record_escalation(expense_id: str, decision: str, reason: str) -> dict:
    ticket_id = f"T{len(STORE.tickets) + 1}"
    notice = f"单据 {ticket_id}：{decision}，{reason}。请等待审批人处理，不要自行入账。"
    ticket = {
        "ticket_id": ticket_id,
        "expense_id": expense_id,
        "decision": decision,
        "reason": reason,
        "notice": notice,
    }
    STORE.tickets.append(ticket)
    return ticket


@tool(name="lookup_policy_text", description="返回报销制度原文，不根据问题改写限额")
def lookup_policy_text(topic: str = "") -> dict:
    clauses = list(POLICY_CLAUSES)
    catalog = current_catalog()
    if catalog is not None and catalog.has("rerank"):
        try:
            ranked = catalog.reranker().rerank(topic or "报销制度", clauses)
            ordered = [
                clauses[item["index"]]
                for item in sorted(ranked, key=lambda item: item["score"], reverse=True)
                if 0 <= item["index"] < len(clauses)
            ]
            if ordered:
                clauses = ordered
        except ConfigurationError:
            clauses = list(POLICY_CLAUSES)
    return {"topic": topic, "text": "".join(clauses)}
