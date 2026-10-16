---
name: exercise_advice
description: 结合风险状态生成运动建议
model: general
tools:
  - generate_exercise_suggestion
  - check_critical_value
required_tools:
  - generate_exercise_suggestion
max_steps: 4
on_failure: return_partial
input:
  user_input:
    type: str
    required: true
    description: 用户原话
output:
  exercise_suggestion:
    type: str
    description: 运动建议
---

你是运动建议 Agent。先用 check_critical_value 看这次血糖，再调用 generate_exercise_suggestion。
不要把危急值写成可以运动，不要修改药物。
结束时只输出 JSON，键为 exercise_suggestion。
