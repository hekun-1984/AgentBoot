"""执行轨迹追踪。"""

from __future__ import annotations

import time
from typing import Any


class ExecutionTracer:
    """记录一次计划执行中的批次、步骤状态和耗时。"""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def record(self, **event: Any) -> None:
        event.setdefault("timestamp", time.time())
        self._events.append(event)

    def export(self) -> list[dict[str, Any]]:
        return list(self._events)
