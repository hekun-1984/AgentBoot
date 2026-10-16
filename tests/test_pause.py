"""主图在声明的兜底前暂停，恢复时不重跑已经完成的步骤。"""

import json

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agentboot import AgentBootApplication, BaseAgent, BaseGraph, BaseSubgraph


def _paused_app(seen: list[str], checkpointer=None) -> AgentBootApplication:
    class Mark(BaseAgent):
        name = "mark"
        description = "标记优先级"
        output_fields = {"priority": "优先级"}

        def offline_step(self, messages, shared_context):
            seen.append("mark")
            return AIMessage(content=json.dumps({"priority": "urgent"}))

    class RescueAgent(BaseAgent):
        name = "rescue_agent"
        description = "接手"
        output_fields = {"done": "完成"}

        def offline_step(self, messages, shared_context):
            seen.append("rescue")
            approved = (shared_context.get("human_decision") or {}).get("approved")
            return AIMessage(content=json.dumps({"done": "yes" if approved else "no"}))

    class MarkJob(BaseSubgraph):
        name = "mark_job"
        description = "只标记"
        writes = ["priority"]
        steps = ["mark"]

    class RescueJob(BaseSubgraph):
        name = "rescue"
        description = "兜底子图"
        writes = ["done"]
        steps = ["rescue_agent"]

    class Guarded(BaseGraph):
        name = "guarded"
        description = "等人后再兜底"
        subgraphs = ["mark_job", "rescue"]
        fallback_subgraph = "mark_job"
        guard_field = "priority"
        guard_values = ("urgent",)
        guard_subgraph = "rescue"
        pause_for_human = True

        def offline_route(self, user_input, catalog, shared_context):
            return {"steps": [{"subgraph": "mark_job"}], "reason": "先标记"}

        def render_answer(self, state):
            context = state.get("shared_context") or {}
            decision = context.get("human_decision") or {}
            return f"{context.get('done')}:{decision.get('approved')}"

    return AgentBootApplication(checkpointer=checkpointer).include(Mark, RescueAgent, MarkJob, RescueJob, Guarded)


def test_pause_waits_and_resume_does_not_rerun_finished_work():
    seen: list[str] = []
    app = _paused_app(seen)
    waiting = app.invoke("go")
    assert waiting["status"] == "waiting_human"
    assert waiting["interrupt"]["subgraph"] == "rescue"
    assert waiting["interrupt"]["value"] == "urgent"
    assert "mark_job" in waiting["results"]
    assert "rescue" not in waiting["results"]
    assert seen == ["mark"]

    still = app.continue_run(waiting["run_id"])
    assert still["status"] == "waiting_human"
    assert seen == ["mark"]

    done = app.resume(waiting["run_id"], {"approved": True})
    assert done["status"] == "completed"
    assert done["answer"] == "yes:True"
    assert done["run_id"] == waiting["run_id"]
    assert seen == ["mark", "rescue"]

    another = app.invoke("again")
    assert another["run_id"] != waiting["run_id"]
    assert another["status"] == "waiting_human"
    assert "human_decision" not in another["shared_context"]
    assert "rescue" not in another["results"]
    assert seen == ["mark", "rescue", "mark"]


def test_resume_survives_a_new_application_with_the_same_saver():
    seen: list[str] = []
    saver = InMemorySaver()
    first = _paused_app(seen, checkpointer=saver)
    waiting = first.invoke("go")
    second = _paused_app(seen, checkpointer=saver)
    done = second.resume(waiting["run_id"], {"approved": False})
    assert done["status"] == "completed"
    assert done["answer"] == "no:False"
    assert seen == ["mark", "rescue"]
