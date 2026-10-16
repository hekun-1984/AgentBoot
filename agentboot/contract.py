"""用 SKILL 声明校验输出。缺字段或类型不对时，调用方按失败策略处理。"""

from __future__ import annotations

from agentboot.models import FieldSpec


def value_matches(value: object, type_name: str) -> bool:
    kind = (type_name or "str").lower()
    if kind in {"str", "string"}:
        return isinstance(value, str)
    if kind in {"bool", "boolean"}:
        return isinstance(value, bool)
    if kind in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind in {"dict", "object", "map"}:
        return isinstance(value, dict)
    if kind in {"list", "array"}:
        return isinstance(value, list)
    return True


def output_problems(data: dict, outputs: dict[str, FieldSpec]) -> list[str]:
    problems: list[str] = []
    for name, field in outputs.items():
        if name not in data or data[name] is None:
            if field.required:
                problems.append(f"缺少字段 {name}")
            continue
        if not value_matches(data[name], field.type_name):
            problems.append(f"字段 {name} 类型应为 {field.type_name}")
    return problems
