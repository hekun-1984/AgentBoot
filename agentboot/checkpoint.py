"""同一次流程的存档。表结构用 LangGraph 的 Postgres checkpointer，框架不另建表。"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from agentboot.exceptions import ConfigurationError

DURABILITY = "sync"


def wait_for_human(payload: Any) -> Any:
    """在当前步骤停下等人。返回值是 resume 时提交的决定或补充信息。

    可以写在 Tool 函数、Agent 节点或子图节点里。副作用放在这次调用返回之后，
    因为暂停后这一步会从头再执行到这里。
    """
    return interrupt(payload)


def open_postgres_checkpointer(conn_string: str) -> Any:
    """打开 LangGraph 的 Postgres 存档，并创建它自己的表。

    `PostgresSaver.setup()` 负责这四张表：`checkpoint_migrations`、`checkpoints`、
    `checkpoint_blobs`、`checkpoint_writes`。连接要一直留着，返回的对象传给
    `AgentBootApplication(checkpointer=...)`。
    """
    try:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise ConfigurationError(
            "Postgres 存档需要安装 langgraph-checkpoint-postgres 和 psycopg。"
            "表由 LangGraph 的 PostgresSaver.setup() 创建，框架不定义自己的表。"
        ) from exc
    connection = psycopg.connect(conn_string, autocommit=True)
    saver = PostgresSaver(connection)
    saver.setup()
    return saver
