"""只引用制度原文。空计划时主图落到这里。"""

from agentboot import BaseSubgraph


class PolicyReferenceSubgraph(BaseSubgraph):
    name = "policy_reference"
    description = "返回报销制度原文，不审批具体单据。"
    reads = ["user_input"]
    writes = ["policy_text"]
    steps = ["policy_lookup"]
