"""子图、Agent、Tool 落在同一次流程的存档里，中断后不重跑已经完成的步骤。"""

import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agentboot import AgentBootApplication, BaseAgent, BaseGraph, BaseSubgraph, tool, wait_for_human
from agentboot.checkpoint import open_postgres_checkpointer
from agentboot.exceptions import AgentExecutionError, ConfigurationError


def test_finished_tool_is_not_rerun_when_a_later_agent_crashes():
    seen: list[str] = []

    @tool(name="mark_done", description="记下已经完成的一步")
    def mark_done() -> dict:
        seen.append("tool")
        return {"flag": "set"}

    class Later(BaseAgent):
        name = "later"
        description = "后一步"
        output_fields = {"flag": "标记"}

        def offline_step(self, messages, shared_context):
            seen.append("agent")
            if seen.count("agent") == 1:
                raise AgentExecutionError("后一步中断", error_kind="llm", agent="later")
            return AIMessage(content=json.dumps({"flag": "done"}))

    class Job(BaseSubgraph):
        name = "job"
        description = "先工具再代理"
        writes = ["flag"]

        def build(self, builder, runtime):
            runtime.add_tool(builder, "mark", "mark_done", read={}, write={"flag": "flag"})
            runtime.add_agent(builder, "later")
            from langgraph.graph import END, START

            builder.add_edge(START, "mark")
            builder.add_edge("mark", "later")
            builder.add_edge("later", END)

    class Line(BaseGraph):
        name = "line"
        description = "一条流程"
        subgraphs = ["job"]
        fallback_subgraph = "job"

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [{"subgraph": "job"}], "reason": "固定"}

        def render_answer(self, state):
            return str((state.get("shared_context") or {}).get("flag") or "")

    app = AgentBootApplication(checkpointer=InMemorySaver()).include(mark_done, Later, Job, Line)
    with pytest.raises(AgentExecutionError):
        app.invoke("go", run_id="case-1")
    assert seen == ["tool", "agent"]

    done = app.continue_run("case-1")
    assert done["status"] == "completed"
    assert done["answer"] == "done"
    assert seen == ["tool", "agent", "agent"]


def test_tool_can_wait_for_a_person_and_then_continue():
    seen: list[int] = []

    @tool(name="settle", description="等人确认后再入账")
    def settle(amount: int) -> dict:
        decision = wait_for_human({"message": "确认入账", "amount": amount})
        approved = isinstance(decision, dict) and bool(decision.get("approved"))
        if not approved:
            return {"posted": "no"}
        seen.append(amount)
        return {"posted": "yes", "note": str(decision.get("note") or "")}

    class Pay(BaseSubgraph):
        name = "pay"
        description = "入账"
        writes = ["posted", "note"]

        def build(self, builder, runtime):
            runtime.add_tool(
                builder,
                "settle",
                "settle",
                read={"amount": "amount"},
                write={"posted": "posted", "note": "note"},
            )
            from langgraph.graph import END, START

            builder.add_edge(START, "settle")
            builder.add_edge("settle", END)

    class Line(BaseGraph):
        name = "pay_line"
        description = "入账流程"
        subgraphs = ["pay"]
        fallback_subgraph = "pay"

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [{"subgraph": "pay"}], "reason": "固定"}

        def render_answer(self, state):
            context = state.get("shared_context") or {}
            return f"{context.get('posted')}:{context.get('note')}"

    saver = InMemorySaver()
    app = AgentBootApplication(checkpointer=saver).include(settle, Pay, Line)
    waiting = app.invoke("go", {"amount": 80}, run_id="pay-1")
    assert waiting["status"] == "waiting_human"
    assert waiting["interrupt"]["amount"] == 80
    assert seen == []

    held = app.continue_run("pay-1")
    assert held["status"] == "waiting_human"
    assert seen == []

    done = AgentBootApplication(checkpointer=saver).include(settle, Pay, Line).resume(
        "pay-1",
        {"approved": True, "note": "补了发票"},
    )
    assert done["status"] == "completed"
    assert done["answer"] == "yes:补了发票"
    assert seen == [80]


def test_postgres_helper_asks_for_langgraph_saver(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fail_postgres(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psycopg" or name.startswith("langgraph.checkpoint.postgres"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_postgres)
    with pytest.raises(ConfigurationError, match="PostgresSaver"):
        open_postgres_checkpointer("postgresql://localhost/agentboot")
