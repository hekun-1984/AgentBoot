"""回答之后必须过合规校验，没有来源或不在范围内都不能直接返回。"""

from langgraph.graph import END, START

from agentboot import BaseSubgraph


class DiabetesGuidelineQASubgraph(BaseSubgraph):
    name = "diabetes_guideline_qa"
    description = "依据糖尿病指南回答问题。超出范围时拒绝，回答必须附带来源。"
    reads = ["user_input"]
    writes = ["guideline_answer", "sources", "in_scope", "compliance_passed"]

    def build(self, builder, runtime):
        runtime.add_agent(builder, "guideline_qa")
        runtime.add_tool(
            builder,
            "compliance",
            "check_guideline_compliance",
            read={"answer": "guideline_answer", "in_scope": "in_scope", "sources_json": "sources"},
            write={
                "guideline_answer": "answer",
                "compliance_passed": "passed",
                "sources": "sources",
            },
            json_args=("sources_json",),
        )
        builder.add_edge(START, "guideline_qa")
        builder.add_edge("guideline_qa", "compliance")
        builder.add_edge("compliance", END)
