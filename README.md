# AgentBoot
AgentBoot: An out-of-the-box Agent business development framework inspired by Spring Boot. Register subgraphs or capabilities to start building. | 借鉴 Spring Boot 体验的 Agent 业务开发框架，注册子图/能力即可二次开发。

内部实现基于 LangGraph 的 Tool、Subgraph 和 StateGraph；对外抽象为 Capability、SkillGraph 和 OrchestratorGraph。

Capability 层    → 原子能力，类似 DAO
SkillGraph 层    → 固定流程子图，类似 Service
OrchestratorGraph 层 → 主图动态路由，类似 Controller