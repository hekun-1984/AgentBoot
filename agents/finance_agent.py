"""示例：财务分析 Agent。展示多批次依赖：采集、并行分析、汇总报告。"""

from __future__ import annotations

from typing import Any

from agentboot.graph import BaseSubGraph
from agentboot.models import ExecutionPlan
from agents.base_agent import BaseAgent
from plans.finance_plan import FinancePlan
from skills.finance_skill import DataQuery, FinanceAnalysis, ReportGenerate, RiskAssessment


class MarketDataFetcher(BaseSubGraph):
    """子图：获取账户与财务输入。"""

    name = "market_data_fetch"
    description = "读取账户持仓和收入成本"
    parallel_group = "data_collection"
    allowed_skills = ["data_query"]

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        return await self.call_skill("data_query", {"account_id": context.get("account_id")})


class StatementAnalyzer(BaseSubGraph):
    """子图：财报分析。与风险评估并行。"""

    name = "statement_analyzer"
    description = "计算利润和利润率"
    depends_on = ["market_data_fetch"]
    parallel_group = "analysis_phase"
    allowed_skills = ["finance_analysis"]

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        fetched = context["market_data_fetch"]
        analysis = await self.call_skill(
            "finance_analysis",
            {"revenue": fetched["revenue"], "cost": fetched["cost"]},
        )
        return {"statement": analysis}


class PortfolioRisk(BaseSubGraph):
    """子图：组合风险。与财报分析并行。"""

    name = "portfolio_risk"
    description = "评估持仓集中度和波动率"
    depends_on = ["market_data_fetch"]
    parallel_group = "analysis_phase"
    allowed_skills = ["risk_assessment"]

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        fetched = context["market_data_fetch"]
        risk = await self.call_skill("risk_assessment", {"positions": fetched["positions"]})
        return {"risk": risk}


class ReportWriter(BaseSubGraph):
    """子图：汇总报告。依赖财报分析和风险评估。"""

    name = "report_writer"
    description = "生成财务简报"
    depends_on = ["statement_analyzer", "portfolio_risk"]
    allowed_skills = ["report_generate"]

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        fetched = context["market_data_fetch"]
        report = await self.call_skill(
            "report_generate",
            {
                "account_id": fetched["account_id"],
                "statement": context["statement_analyzer"]["statement"],
                "risk": context["portfolio_risk"]["risk"],
            },
        )
        return report


class FinanceAgent(BaseAgent):
    """财务分析 Agent。"""

    name = "finance_agent"
    description = "财务分析助手：数据采集 -> 财报分析与风险评估 -> 简报"

    def __init__(self) -> None:
        super().__init__()
        self.register_skill(DataQuery)
        self.register_skill(FinanceAnalysis)
        self.register_skill(RiskAssessment)
        self.register_skill(ReportGenerate)
        self.register_subgraph(MarketDataFetcher)
        self.register_subgraph(StatementAnalyzer)
        self.register_subgraph(PortfolioRisk)
        self.register_subgraph(ReportWriter)

    def sample_context(self) -> dict[str, Any]:
        return {"account_id": "A100"}

    async def execute_plan(self, context: dict[str, Any]) -> ExecutionPlan:
        return FinancePlan().build(context)
