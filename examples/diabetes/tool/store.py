"""示例诊所数据。Tool 只读写这里，方便替换成真实仓储。"""

from __future__ import annotations

import threading


class ClinicStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.patients = {
            "p001": {
                "patient_id": "p001",
                "name": "张三",
                "age": 56,
                "sex": "男",
                "diagnoses": ["2型糖尿病", "高血压"],
                "allergies": [],
                "history": ["2型糖尿病 8 年"],
                "preferences": {"diet": "清淡", "exercise": "散步"},
            },
            "p002": {
                "patient_id": "p002",
                "name": "李四",
                "age": 61,
                "sex": "女",
                "diagnoses": ["2型糖尿病"],
                "allergies": ["青霉素"],
                "history": ["反复低血糖"],
                "preferences": {"diet": "少油", "exercise": "散步"},
            },
        }
        self.glucose = {
            "p001": [
                {"date": "2026-10-13", "period": "fasting", "value": 7.8, "unit": "mmol/L"},
                {"date": "2026-10-14", "period": "fasting", "value": 7.6, "unit": "mmol/L"},
                {"date": "2026-10-15", "period": "fasting", "value": 7.9, "unit": "mmol/L"},
            ],
            "p002": [
                {"date": "2026-10-15", "period": "random", "value": 2.5, "unit": "mmol/L"},
            ],
        }
        self.medications = {
            "p001": [{"name": "二甲双胍", "dose": "0.5g", "frequency": "每日两次"}],
            "p002": [{"name": "二甲双胍", "dose": "0.5g", "frequency": "每日两次"}],
        }
        self.events: list[dict] = []
        self.reminders: list[dict] = []

    def reset_activity(self) -> None:
        with self._lock:
            self.events.clear()
            self.reminders.clear()


STORE = ClinicStore()
