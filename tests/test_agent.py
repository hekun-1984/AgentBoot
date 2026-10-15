import json

import pytest
from langchain_core.messages import AIMessage

from agentboot import BaseAgent, ScriptedModel, tool
from agentboot.exceptions import AgentExecutionError, RegistrationError, SkillError
from agentboot.skill import load_skill


def test_skill_contract_declares_tools_steps_and_failure(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        """---
name: demo
description: 演示契约
tools:
  - ping
max_steps: 3
on_failure: escalate
input:
  patient_id:
    type: str
    required: true
    description: 患者
output:
  text:
    type: str
    description: 结果
---

只做一件事。
""",
        encoding="utf-8",
    )
    loaded = load_skill(skill)
    assert loaded.tools == ["ping"]
    assert loaded.max_steps == 3
    assert loaded.on_failure == "escalate"
    assert loaded.inputs["patient_id"].required is True
    assert loaded.system_prompt == "只做一件事。"


def test_duplicate_tool_name_is_rejected():
    @tool(name="same", description="a")
    def first(value: str) -> dict:
        return {"value": value}

    @tool(name="same", description="b")
    def second(value: str) -> dict:
        return {"value": value}

    from agentboot.registry import Registry

    registry = Registry()
    registry.add(first)
    with pytest.raises(RegistrationError):
        registry.add(second)


def test_agent_runs_tool_loop_and_records_child_spans():
    seen = {}

    @tool(name="ping", description="回声")
    def ping(value: str) -> dict:
        seen["value"] = value
        return {"value": value}

    class EchoAgent(BaseAgent):
        name = "echo"
        description = "回声"
        tools = ["ping"]
        max_steps = 4
        on_failure = "return_partial"
        output_fields = {"text": "文本"}

        def offline_step(self, messages, shared_context):
            raise AssertionError("单测应走脚本模型")

    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[{"name": "ping", "args": {"value": "pong"}, "id": "call-1"}]),
            AIMessage(content=json.dumps({"text": "pong"})),
        ]
    )
    result = EchoAgent().bind({"ping": ping.__agentboot_tool__}, model).invoke(
        {"request_id": "r1"},
        user_input="ping",
    )
    assert result.output["text"] == "pong"
    assert seen["value"] == "pong"
    assert result.error is None
    assert result.trace[0]["kind"] == "agent"
    assert result.trace[0]["state_before"]["request_id"] == "r1"
    assert {child["kind"] for child in result.trace[0]["children"]} == {"llm", "tool"}
    assert model.calls == 2


def test_unknown_tool_is_an_llm_error_and_is_not_executed():
    called = {"value": False}

    @tool(name="ping", description="回声")
    def ping(value: str) -> dict:
        called["value"] = True
        return {"value": value}

    class Guarded(BaseAgent):
        name = "guarded"
        description = "白名单"
        tools = ["ping"]
        max_steps = 3
        output_fields = {"text": "文本"}

    model = ScriptedModel(
        [AIMessage(content="", tool_calls=[{"name": "other", "args": {}, "id": "x"}])]
    )
    result = Guarded().bind({"ping": ping.__agentboot_tool__}, model).invoke({}, user_input="go")
    assert result.error_kind == "llm"
    assert called["value"] is False


def test_max_steps_stops_the_tool_loop():
    @tool(name="ping", description="回声")
    def ping(value: str) -> dict:
        return {"value": value}

    class Limited(BaseAgent):
        name = "limited"
        description = "步数"
        tools = ["ping"]
        max_steps = 1
        on_failure = "return_partial"
        output_fields = {"text": "文本"}

    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[{"name": "ping", "args": {"value": "a"}, "id": "1"}]),
            AIMessage(content="", tool_calls=[{"name": "ping", "args": {"value": "b"}, "id": "2"}]),
        ]
    )
    result = Limited().bind({"ping": ping.__agentboot_tool__}, model).invoke({}, user_input="go")
    assert result.error_kind == "llm"
    assert "最大步数" in result.error
    assert model.calls == 1


def test_tool_exception_is_classified_as_tool_error():
    @tool(name="boom", description="失败")
    def boom(value: str) -> dict:
        raise RuntimeError("下游不可用")

    class ToolAgent(BaseAgent):
        name = "tool_agent"
        description = "工具失败"
        tools = ["boom"]
        max_steps = 3
        output_fields = {"text": "文本"}

    model = ScriptedModel(
        [AIMessage(content="", tool_calls=[{"name": "boom", "args": {"value": "x"}, "id": "1"}])]
    )
    result = ToolAgent().bind({"boom": boom.__agentboot_tool__}, model).invoke({}, user_input="go")
    assert result.error_kind == "tool"
    kinds = [child["error_kind"] for child in result.trace[0]["children"]]
    assert "tool" in kinds


def test_missing_required_input_is_a_state_error(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        """---
name: need_id
description: 需要患者
tools: []
max_steps: 2
on_failure: return_partial
input:
  patient_id:
    type: str
    required: true
    description: 患者 ID
output:
  text:
    type: str
    description: 文本
---

需要 patient_id。
""",
        encoding="utf-8",
    )

    class NeedId(BaseAgent):
        name = "need_id"
        skill_path = str(skill)

        def offline_step(self, messages, shared_context):
            return AIMessage(content=json.dumps({"text": "ok"}))

    result = NeedId().bind({}).invoke({}, user_input="有话但没有患者")
    assert result.error_kind == "state"
    assert result.steps == 0


def test_abort_policy_raises():
    @tool(name="ping", description="回声")
    def ping(value: str) -> dict:
        return {"value": value}

    class Strict(BaseAgent):
        name = "strict"
        description = "中断"
        tools = ["ping"]
        on_failure = "abort"
        output_fields = {"text": "文本"}

    model = ScriptedModel([AIMessage(content="不是 json")])
    agent = Strict().bind({"ping": ping.__agentboot_tool__}, model)
    with pytest.raises(AgentExecutionError) as caught:
        agent.invoke({}, user_input="go")
    assert caught.value.error_kind == "llm"


def test_skill_name_must_match_class(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: other\ndescription: x\ntools: []\n---\n",
        encoding="utf-8",
    )

    class Mismatch(BaseAgent):
        name = "mismatch"
        skill_path = str(skill)

    with pytest.raises(SkillError):
        Mismatch()
