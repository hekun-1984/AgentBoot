# AgentBoot

AgentBoot: An out-of-the-box Agent business development framework inspired by Spring Boot. Register subgraphs or capabilities to start building. | 借鉴 Spring Boot 体验的 Agent 业务开发框架，注册子图/能力即可二次开发。

内部编排不绑定具体图框架。默认使用内置并行调度器；设置 `AGENTBOOT_ENGINE=langgraph` 后，同一份执行计划会编译为 LangGraph 的 StateGraph。对外仍是三层抽象：

| 抽象 | 代码 | 角色 |
| --- | --- | --- |
| Capability | `skills.base_skill.BaseSkill` | 原子能力，类似 DAO |
| SkillGraph | `agentboot.graph.BaseSubGraph` | 固定流程子图，类似 Service |
| OrchestratorGraph | `agents.base_agent.BaseAgent` | 主图动态路由，类似 Controller |

一次调用的路径是：Agent 或 LLM 生成 `ExecutionPlan`，校验器检查依赖图，路由器核对能力白名单，调度器按拓扑分批并行执行，并留下轨迹。

## 快速开始

```bash
pip install -e ".[dev]"
python main.py
```

`main.py` 会注册示例里的糖尿病管理 Agent 和财务分析 Agent，并各跑一遍虚构数据。医疗和财务结果只用于演示框架，不能用于临床或投资决策。

未配置 `AGENTBOOT_LLM_API_KEY` 时，框架使用 `plans/` 中的计划模板。配置 Key 后，调用时传入 `intent` 会先请求模型生成计划，失败再回退到模板。复制 `.env.example` 为 `.env` 即可修改模型、引擎和是否运行示例。

## 二次开发

1. 在 `skills/` 实现 `BaseSkill`。
2. 在 `agents/` 用 `BaseSubGraph` 写出固定流程，并在 Agent 构造函数里注册。
3. 在 `plans/` 提供计划模板，或实现 `execute_plan`。
4. 在 `routes/route_config.py` 写入该 Agent 允许调用的能力。
5. 在 `main.py` 调用 `app.register_agent(...)`。

子图用 `depends_on` 声明前置步骤，用 `parallel_group` 声明可以并行的阶段。同一组里的步骤不能互相依赖。调度器按依赖分批，同一批里的步骤并发执行；任一步失败后，后续批次标记为跳过。
