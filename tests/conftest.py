"""测试夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentboot.app import AgentApp
from agents.diabetes_agent import DiabetesAgent
from agents.finance_agent import FinanceAgent


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AgentApp:
    monkeypatch.setenv("AGENTBOOT_ENGINE", "native")
    monkeypatch.delenv("AGENTBOOT_LLM_API_KEY", raising=False)
    application = AgentApp(app_name="test", project_root=ROOT)
    application.register_agent(DiabetesAgent)
    application.register_agent(FinanceAgent)
    application.bootstrap()
    return application
