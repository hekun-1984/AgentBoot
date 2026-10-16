"""示例里的报销单和升级单。数据是固定的，方便回归。"""

from __future__ import annotations


class ExpenseStore:
    def __init__(self) -> None:
        self.claims = {
            "e100": {
                "expense_id": "e100",
                "applicant": "周宁",
                "category": "办公",
                "amount": 800.0,
                "title": "打印纸",
            },
            "e200": {
                "expense_id": "e200",
                "applicant": "周宁",
                "category": "招待",
                "amount": 80000.0,
                "title": "客户宴请",
            },
        }
        self.tickets: list[dict] = []

    def reset_activity(self) -> None:
        self.tickets.clear()


STORE = ExpenseStore()
