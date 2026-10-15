"""示例：金融领域可复用能力。数据是虚构的，不能用于投资决策。"""

from __future__ import annotations

from typing import Any

from skills.base_skill import BaseSkill

DISCLAIMER = "演示数据，不能用于投资决策。"

ACCOUNTS: dict[str, dict[str, Any]] = {
    "A100": {
        "account_id": "A100",
        "positions": [
            {"symbol": "AAA", "weight": 0.72, "volatility": 0.28},
            {"symbol": "BBB", "weight": 0.28, "volatility": 0.12},
        ],
        "revenue": 120.0,
        "cost": 86.0,
    },
    "A200": {
        "account_id": "A200",
        "positions": [
            {"symbol": "CCC", "weight": 0.25, "volatility": 0.08},
            {"symbol": "DDD", "weight": 0.25, "volatility": 0.08},
            {"symbol": "EEE", "weight": 0.25, "volatility": 0.07},
            {"symbol": "FFF", "weight": 0.25, "volatility": 0.07},
        ],
        "revenue": 80.0,
        "cost": 50.0,
    },
}


def _account(context: dict[str, Any]) -> dict[str, Any]:
    account_id = str(context.get("account_id") or "")
    record = ACCOUNTS.get(account_id)
    if record is None:
        raise ValueError(f"未知账户: {account_id}")
    return record


def assess_portfolio(positions: list[dict[str, Any]]) -> dict[str, Any]:
    if not positions:
        raise ValueError("持仓为空，无法评估。")
    max_weight = max(float(item["weight"]) for item in positions)
    volatility = sum(float(item["weight"]) * float(item["volatility"]) for item in positions)
    if max_weight >= 0.6 or volatility >= 0.25:
        level = "high"
        advice = "单一持仓权重过高，建议降低集中度。"
    elif max_weight >= 0.4 or volatility >= 0.15:
        level = "medium"
        advice = "组合波动中等，建议复核行业暴露。"
    else:
        level = "low"
        advice = "持仓分散，维持再平衡规则即可。"
    return {
        "risk_level": level,
        "max_weight": round(max_weight, 4),
        "volatility": round(volatility, 4),
        "advice": advice,
        "disclaimer": DISCLAIMER,
    }


class DataQuery(BaseSkill):
    """账户与行情数据查询。"""

    name = "data_query"
    description = "查询账户持仓和财务报表输入"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        record = _account(context)
        return {
            "account_id": record["account_id"],
            "positions": list(record["positions"]),
            "revenue": record["revenue"],
            "cost": record["cost"],
            "disclaimer": DISCLAIMER,
        }


class FinanceAnalysis(BaseSkill):
    """财报分析能力。"""

    name = "finance_analysis"
    description = "计算收入、成本和利润率"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        revenue = float(context.get("revenue") or 0)
        cost = float(context.get("cost") or 0)
        profit = revenue - cost
        margin = profit / revenue if revenue else 0.0
        return {
            "profit": round(profit, 2),
            "margin": round(margin, 4),
            "summary": f"利润 {profit:.2f}，利润率 {margin:.1%}。",
            "disclaimer": DISCLAIMER,
        }


class RiskAssessment(BaseSkill):
    """组合风险评估。"""

    name = "risk_assessment"
    description = "根据持仓集中度和波动率评估组合风险"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        return assess_portfolio(list(context.get("positions") or []))


class ReportGenerate(BaseSkill):
    """报告生成能力。"""

    name = "report_generate"
    description = "把财报分析和风险评估合成一份简报"

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        statement = context.get("statement") or {}
        risk = context.get("risk") or {}
        account_id = context.get("account_id", "")
        text = (
            f"账户 {account_id}：{statement.get('summary', '')}"
            f"组合风险 {risk.get('risk_level', 'unknown')}，{risk.get('advice', '')}"
        )
        return {"report": text, "disclaimer": DISCLAIMER}
