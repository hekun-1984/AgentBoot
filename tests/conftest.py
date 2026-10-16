import pytest

from agentboot.catalog import use_catalog
from diabetes.tool.store import STORE as DIABETES_STORE
from expense.tool.store import STORE as EXPENSE_STORE


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    monkeypatch.delenv("AGENTBOOT_MODEL", raising=False)
    monkeypatch.delenv("AGENTBOOT_MODELS", raising=False)
    use_catalog(None)
    DIABETES_STORE.reset_activity()
    EXPENSE_STORE.reset_activity()
    yield
    use_catalog(None)
    DIABETES_STORE.reset_activity()
    EXPENSE_STORE.reset_activity()
