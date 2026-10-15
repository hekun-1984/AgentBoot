"""AgentBoot 公共 API。

对外三层抽象：
- Capability：skills.base_skill.BaseSkill，原子能力，类似 DAO
- SkillGraph：BaseSubGraph，固定流程子图，类似 Service
- OrchestratorGraph：agents.base_agent.BaseAgent，主图动态路由，类似 Controller

默认用内置调度器执行。AGENTBOOT_ENGINE=langgraph 时编译为 LangGraph StateGraph。
"""

from agentboot.app import AgentApp
from agentboot.graph import BaseSubGraph
from agentboot.models import ExecutionPlan, ExecutionResult, Step, StepStatus

SkillGraph = BaseSubGraph

__version__ = "0.1.0"

__all__ = [
    "AgentApp",
    "BaseSubGraph",
    "ExecutionPlan",
    "ExecutionResult",
    "SkillGraph",
    "Step",
    "StepStatus",
    "__version__",
]
