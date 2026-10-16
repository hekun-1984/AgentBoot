from expense import build_app
from expense.tool.store import STORE


def test_office_expense_passes_without_escalation():
    result = build_app().invoke("这张办公费能报销吗？", {"expense_id": "e100"})
    assert result["shared_context"]["decision"] == "pass"
    assert result["shared_context"]["policy_reason"] == "金额在制度限额内"
    assert result["results"].keys() == {"expense_review"}
    assert STORE.tickets == []
    assert "审批结论：pass" in result["answer"]


def test_entertainment_over_limit_uses_declared_guard():
    result = build_app().invoke("这张招待费请审批", {"expense_id": "e200"})
    assert result["shared_context"]["decision"] == "escalate"
    assert result["shared_context"]["policy_reason"] == "招待费超过五千元，需要升级审批"
    assert "expense_review" in result["results"]
    assert "expense_escalation" in result["results"]
    assert "字段兜底:decision=escalate" in result["route_reason"]
    assert STORE.tickets[0]["expense_id"] == "e200"
    assert "不要自行入账" in result["answer"]


def test_unknown_question_falls_back_to_policy_text():
    result = build_app().invoke("今天天气怎么样？", {})
    assert result["plan"]["steps"][0]["subgraph"] == "policy_reference"
    assert "空计划兜底" in result["route_reason"]
    assert "招待费超过五千元" in result["answer"]
    assert "expense_review" not in result["results"]


def test_review_refuses_to_start_without_expense_id():
    blocked = build_app().invoke_subgraph("expense_review", {}, user_input="帮我报销")
    assert blocked["errors"][0]["error_kind"] == "state"
    assert "expense_id" in blocked["errors"][0]["message"]


def test_policy_text_is_reranked_when_the_slot_is_configured():
    from agentboot import ModelCatalog
    from agentboot.catalog import RerankClient, use_catalog
    from expense.agent.policy_check.agent import PolicyCheckAgent
    from expense.tool.definitions import lookup_policy_text

    class Rank:
        def rerank(self, query, documents):
            last = len(documents) - 1
            return [{"index": last, "score": 1.0}] + [{"index": index, "score": 0.0} for index in range(last)]

    use_catalog(ModelCatalog().bind("rerank", RerankClient([("ranker", Rank())])))
    text = lookup_policy_text("报销限额")["text"]
    assert text.startswith("超过十万元不予通过")
    assert "招待费超过五千元" in text
    assert PolicyCheckAgent().spec.model_slot == "reason"
