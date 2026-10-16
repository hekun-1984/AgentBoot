"""只读取报销单，不判断限额。"""

from pathlib import Path

from agentboot import BaseAgent

from expense.agent.common import calls, data_of, final, payloads


class ExpenseReaderAgent(BaseAgent):
    name = "expense_reader"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        if "get_expense" not in found:
            return calls(("get_expense", {"expense_id": str(shared_context.get("expense_id") or "")}, "expense"))
        data = data_of(found, "get_expense")
        amount = data.get("amount")
        return final(
            {
                "amount": float(amount) if isinstance(amount, (int, float)) else 0.0,
                "category": str(data.get("category") or ""),
                "title": str(data.get("title") or ""),
            }
        )
