"""Subgraph 层。每张子图一个文件，边在文件里写死。"""

from diabetes.subgraph.emergency import EmergencyEscalationSubgraph
from diabetes.subgraph.guideline_qa import DiabetesGuidelineQASubgraph
from diabetes.subgraph.management_plan import DiabetesManagementPlanSubgraph
from diabetes.subgraph.risk_check import DiabetesRiskCheckSubgraph

__all__ = [
    "DiabetesGuidelineQASubgraph",
    "DiabetesManagementPlanSubgraph",
    "DiabetesRiskCheckSubgraph",
    "EmergencyEscalationSubgraph",
]
