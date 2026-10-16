"""超限后只登记升级单。"""

from agentboot import BaseSubgraph


class ExpenseEscalationSubgraph(BaseSubgraph):
    name = "expense_escalation"
    description = "把已经判定为升级或拒绝的单据登记给审批人。"
    reads = ["expense_id", "user_input"]
    writes = ["ticket_id", "notice"]
    steps = ["escalation_clerk"]
