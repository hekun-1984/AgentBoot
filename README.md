# AgentBoot

基于 LangGraph 的 Agent 业务开发框架。注册 Tool、Agent、Subgraph，主图就可以编排。

固定流程负责兜底，动态路由只决定这次用哪些能力、按什么顺序、要不要并行。模型在结构化边界里做选择，不负责每个能力内部怎么执行。

## 四层

| 层 | 类比 | 做什么 | 不做什么 |
| --- | --- | --- | --- |
| Tool | DAO | 一次原子读写或规则计算 | 不调用模型，不决定下一步 |
| Agent | 一次模型执行 | 一个模型循环，只能用声明过的 Tool | 不跨子图路由 |
| Subgraph | Service | 用写死的边编排多个 Agent 或必须执行的 Tool | 边不交给模型改 |
| Graph | Controller | 从白名单里选子图，处理并行、步数、空计划和危急兜底 | 不直接乱调 Tool / Agent |

每个 Agent 都是一张独立的 LangGraph，可以单独 `invoke`。输入、输出、工具集、最大步数和失败策略写在该 Agent 目录的 `SKILL.md`。失败会标成 `llm`、`tool` 或 `state`。一次执行的 trace 里，Agent span 下面挂着每次模型调用和 Tool 调用，并带有执行前后的状态快照。

接上真实模型时，结束回答会按 `SKILL.md` 的 `output` 字段调用 `with_structured_output`。类型和缺字段仍由运行时再检查一次。没有声明输出字段、或模型不提供结构化接口时，仍走原来的 JSON 文本。离线 `offline_step` 不走这条路径。

`on_failure` 可以是 `return_partial`、`fallback`、`escalate` 或 `abort`。

## 糖尿病示例

这个示例不是开放闲聊机器人。用户问血糖、饮食、用药、运动、复查或指南时，主图只挑选下面几张子图：

- `diabetes_risk_check`：PatientDataAgent → RiskCheckAgent
- `diabetes_guideline_qa`：GuidelineQAAgent → 合规校验 Tool
- `diabetes_management_plan`：取数 → 饮食 → 运动 → 复查 → 汇总
- `emergency_escalation`：风险评估 → 记录高危事件 → 紧急提示 → 转人工

危急值由 `check_critical_value` 按阈值判断。用户话里出现危急血糖或昏迷、胸痛、手抖等症状时，主图在进路由之前就改走紧急子图，不再执行饮食和运动。子图之间只通过 `shared_context` 里声明的 `writes` 传递数据。

```bash
pip install -e ".[dev]"
python main.py
python main.py 我现在头晕、手抖，刚测血糖 2.9。
pytest
```

未设置模型时，路由和 Agent 使用可重复的离线实现，测试不需要密钥。要换真实模型：

```bash
set AGENTBOOT_MODEL=openai:gpt-4o-mini
python main.py
```

也可以在代码里把模型只交给某一层。`build_app(model=...)` 同时用于 Agent 和主图路由；`build_app(router=...)` 可以单独替换路由器。安全筛查和高危兜底仍然会执行。

## 二次开发

1. 新建业务包，用 `@tool` 写原子能力。函数签名就是输入，返回值就是输出。
2. 每个 Agent 一个目录：`SKILL.md` 写契约和系统提示词，`agent.py` 继承 `BaseAgent`。没有模型时实现 `offline_step`，有模型时同一张图会改为工具调用循环。
3. 继承 `BaseSubgraph`。顺序流程填 `steps`；中间要插入 Tool 时覆盖 `build`，用 `add_edge` 把边写死。用 `writes` 列出允许写回主图的字段。
4. 继承 `BaseGraph`，在 `subgraphs` 里列出白名单，设 `fallback_subgraph` 和 `max_steps`。需要按状态兜底时声明 `guard_field`、`guard_values` 和 `guard_subgraph`。路由前的规则检查写在 `screen()`。
5. `AgentBootApplication().scan("你的包").invoke(问题, context)`。

单独回归某一个 Agent：

```python
import sys
sys.path.insert(0, "examples")

from diabetes import build_app

app = build_app()
result = app.invoke_agent("risk_check", {"patient_id": "p001", "mentioned_glucose": 2.9}, user_input="刚测血糖 2.9")
print(result.output)
print(result.trace[0]["children"])
```

`result.error_kind` 为 `llm` 表示模型决策或结构化输出失败，`tool` 表示工具抛错，`state` 表示声明的必填输入没有传到。

`pip install` 的发布物只有 `agentboot`。`examples/diabetes` 和 `examples/expense` 留在仓库里，用来演示同一套契约，不进入框架安装包。本地运行和测试会把 `examples` 加进路径。

## 费用审批示例

这个示例不包含医疗规则。限额写在 `check_amount_policy` 里。主图声明 `decision` 为 `escalate` 或 `reject` 时进入 `expense_escalation`，空计划落到 `policy_reference`。

```python
import sys
sys.path.insert(0, "examples")

from expense import build_app

print(build_app().invoke("这张办公费能报销吗？", {"expense_id": "e100"})["answer"])
```

## 目录

`agentboot` 是可安装的框架。示例放在 `examples`，各自再按四层分目录。

```text
agentboot/
├── tool/            # @tool
├── agent/           # BaseAgent
├── subgraph/        # BaseSubgraph
├── graph/           # BaseGraph
├── application.py
└── registry.py
examples/
├── diabetes/        # 医疗示例：档案、血糖、指南、危急值
└── expense/         # 费用审批示例：读单、限额、升级、制度原文
```
