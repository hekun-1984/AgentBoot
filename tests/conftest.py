import pytest

from diabetes.tool.store import STORE


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    monkeypatch.delenv("AGENTBOOT_MODEL", raising=False)
    STORE.reset_activity()
    yield
    STORE.reset_activity()
