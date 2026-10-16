"""制度原文只来自 Tool。"""

from pathlib import Path

from agentboot import BaseAgent

from expense.agent.common import calls, data_of, final, payloads


class PolicyLookupAgent(BaseAgent):
    name = "policy_lookup"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        if "lookup_policy_text" not in found:
            return calls(("lookup_policy_text", {"topic": str(shared_context.get("user_input") or "")}, "text"))
        data = data_of(found, "lookup_policy_text")
        return final({"policy_text": str(data.get("text") or "")})
