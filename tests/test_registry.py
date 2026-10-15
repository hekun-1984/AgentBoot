"""注册中心、路由和配置测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentboot.app import AgentApp
from agentboot.config import Settings
from agentboot.exceptions import ConfigurationError, RegistrationError, RoutingError
from agentboot.router import Router
from agents.diabetes_agent import DiabetesAgent
from skills.medical_skill import PatientRecord

ROOT = Path(__file__).resolve().parents[1]


def test_registry_lists_components(app: AgentApp) -> None:
    assert app.registry.list_agents() == ["diabetes_agent", "finance_agent"]
    assert "patient_data_fetch" in app.registry.list_subgraphs()
    assert "report_writer" in app.registry.list_subgraphs()
    assert "medical_analysis" in app.registry.list_skills()
    assert "data_query" in app.registry.list_skills()
    assert app.registry.get_agent("DiabetesAgent") is DiabetesAgent
    assert app.registry.get_agent_instance("diabetes_agent").name == "diabetes_agent"


def test_duplicate_agent_is_rejected(app: AgentApp) -> None:
    with pytest.raises(RegistrationError):
        app.register_agent(DiabetesAgent)


def test_same_skill_can_be_registered_twice(app: AgentApp) -> None:
    app.register_skill(PatientRecord)
    assert app.registry.get_skill("patient_record") is PatientRecord


def test_router_denies_skill_outside_whitelist(app: AgentApp) -> None:
    with pytest.raises(RoutingError, match="denied"):
        app.router.check_whitelist("diabetes_agent", ["finance_analysis"])


def test_router_allows_declared_skill(app: AgentApp) -> None:
    assert app.router.check_whitelist("finance_agent", ["data_query", "report_generate"]) is True


def test_unknown_agent_is_permissive(tmp_path: Path) -> None:
    router = Router(Settings(project_root=tmp_path))
    router.load_whitelist()
    assert router.check_whitelist("unknown", ["anything"]) is True


def test_settings_env_overrides_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTBOOT_LLM_MODEL", "demo-model")
    monkeypatch.setenv("AGENTBOOT_DEMO", "false")
    monkeypatch.setenv("AGENTBOOT_ENGINE", "native")
    settings = Settings(project_root=tmp_path)
    assert settings.llm_model == "demo-model"
    assert settings.demo is False
    assert settings.engine == "native"


def test_settings_reads_model_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTBOOT_LLM_MODEL", raising=False)
    settings = Settings(project_root=ROOT)
    assert settings.llm_provider == "openai-compatible"
    assert settings.llm_model == "gpt-4o-mini"


def test_invalid_engine_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTBOOT_ENGINE", "nope")
    with pytest.raises(ConfigurationError, match="AGENTBOOT_ENGINE"):
        Settings(project_root=tmp_path)


def test_langgraph_engine_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import importlib.util

    monkeypatch.setenv("AGENTBOOT_ENGINE", "langgraph")
    if importlib.util.find_spec("langgraph") is not None:
        pytest.skip("langgraph is installed")
    with pytest.raises(ConfigurationError, match="langgraph"):
        AgentApp(app_name="lg", project_root=tmp_path)


def test_run_without_demo_returns(app: AgentApp) -> None:
    app.run(demo=False)
