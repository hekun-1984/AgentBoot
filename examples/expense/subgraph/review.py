"""先读单据，再按制度判断。顺序不能交给模型改。"""

from agentboot import BaseSubgraph


class ExpenseReviewSubgraph(BaseSubgraph):
    name = "expense_review"
    description = "读取报销单，并按制度给出通过、升级或拒绝。"
    reads = ["expense_id", "user_input"]
    writes = ["amount", "category", "title", "decision", "policy_reason"]
    steps = ["expense_reader", "policy_check"]
