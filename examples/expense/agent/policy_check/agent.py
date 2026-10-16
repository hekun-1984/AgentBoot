"""限额结论只包装 Tool 的返回。"""

from pathlib import Path

from agentboot import BaseAgent

from expense.agent.common import calls, data_of, final, payloads


class PolicyCheckAgent(BaseAgent):
    name = "policy_check"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        if "check_amount_policy" not in found:
            amount = shared_context.get("amount")
            return calls(
                (
                    "check_amount_policy",
                    {
                        "amount": float(amount) if isinstance(amount, (int, float)) else 0.0,
                        "category": str(shared_context.get("category") or ""),
                    },
                    "policy",
                )
            )
        data = data_of(found, "check_amount_policy")
        return final(
            {
                "decision": str(data.get("decision") or ""),
                "policy_reason": str(data.get("policy_reason") or ""),
            }
        )
