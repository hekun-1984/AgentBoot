---
name: expense_reader
description: 读取报销单上的金额、类别和事由
tools:
  - get_expense
required_tools:
  - get_expense
max_steps: 3
on_failure: return_partial
input:
  expense_id:
    type: str
    required: true
    description: 报销单号
  user_input:
    type: str
    required: true
    description: 用户原话
output:
  amount:
    type: float
    description: 报销金额
  category:
    type: str
    description: 费用类别
  title:
    type: str
    description: 事由
---

你是报销单读取 Agent。金额和类别必须来自 get_expense，不要根据用户原话改数字。
不要判断能不能通过，也不要登记升级单。
