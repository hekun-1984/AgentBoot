"""只依据检索片段回答指南问题。"""

from pathlib import Path

from agentboot import BaseAgent
from diabetes.agents.common import calls, data_of, final, payloads

_TERMS = ("糖尿病", "血糖", "糖化", "hba1c", "胰岛素", "二甲双胍", "低血糖", "饮食", "运动")


class GuidelineQAAgent(BaseAgent):
    name = "guideline_qa"
    skill_path = str(Path(__file__).with_name("SKILL.md"))

    def offline_step(self, messages, shared_context):
        found = payloads(messages)
        question = str(shared_context.get("user_input") or "")
        if "query_guideline_rag" not in found:
            return calls(("query_guideline_rag", {"question": question}, "rag"))
        chunks = list(data_of(found, "query_guideline_rag").get("chunks") or [])
        lowered = question.lower()
        in_scope = bool(chunks) or any(term in lowered for term in _TERMS)
        if not chunks:
            return final({"guideline_answer": "", "sources": [], "in_scope": False})
        return final(
            {
                "guideline_answer": chunks[0]["excerpt"],
                "sources": chunks,
                "in_scope": in_scope,
            }
        )
