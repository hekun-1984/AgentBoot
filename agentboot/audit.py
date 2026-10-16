"""一次调用的审计记录。调试看返回值，留存交给可替换的 TraceSink。"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


_CALL_ID: ContextVar[str] = ContextVar("agentboot_call_id", default="")


def new_call_id() -> str:
    return uuid4().hex


def current_call_id() -> str:
    return _CALL_ID.get()


class TraceSink(Protocol):
    def write(self, record: dict) -> None:
        """留下同一次调用的审计记录。"""


class MemoryTraceSink:
    """进程内记录，测试和当场重放用。"""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(json.loads(json.dumps(record, ensure_ascii=False)))


class JsonlTraceSink:
    """一行一条 JSON，追加写入文件。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


def build_record(*, call_id: str, user_input: str, context: dict, output: dict) -> dict:
    trace = list(output.get("trace") or [])
    return {
        "call_id": call_id,
        "user_input": user_input,
        "context": dict(context),
        "output": output,
        "trace": trace,
    }


def call_scope(call_id: str):
    return _CallScope(call_id)


class _CallScope:
    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self._token: Any = None

    def __enter__(self) -> str:
        self._token = _CALL_ID.set(self.call_id)
        return self.call_id

    def __exit__(self, exc_type, exc, traceback) -> None:
        _CALL_ID.reset(self._token)
