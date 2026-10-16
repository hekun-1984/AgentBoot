"""超限后只登记，不改结论。"""

from pathlib import Path

from agentboot import BaseAgent

from expense.agent.common import calls, data_of, final, payloads


class EscalationClerkAgent(BaseAgent):
    name = "escalation_clerk"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        if "record_escalation" not in found:
            return calls(
                (
                    "record_escalation",
                    {
                        "expense_id": str(shared_context.get("expense_id") or ""),
                        "decision": str(shared_context.get("decision") or ""),
                        "reason": str(shared_context.get("policy_reason") or ""),
                    },
                    "ticket",
                )
            )
        data = data_of(found, "record_escalation")
        return final(
            {
                "ticket_id": str(data.get("ticket_id") or ""),
                "notice": str(data.get("notice") or ""),
            }
        )
