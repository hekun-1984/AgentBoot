"""用 SKILL 声明校验输出，并生成真实模型要用的结构化 schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field, create_model

from agentboot.models import FieldSpec

_PY_TYPES = {
    "str": str,
    "string": str,
    "bool": bool,
    "boolean": bool,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "dict": dict,
    "object": dict,
    "map": dict,
    "list": list,
    "array": list,
}


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


def output_model(agent_name: str, outputs: dict[str, FieldSpec]) -> type[BaseModel]:
    """按 SKILL 的 output 字段生成 Pydantic 模型，交给模型的结构化输出接口。"""
    fields: dict = {}
    for name, spec in outputs.items():
        py_type = _PY_TYPES.get((spec.type_name or "str").lower(), str)
        description = spec.description or name
        if spec.required:
            fields[name] = (py_type, Field(description=description))
        else:
            fields[name] = (py_type | None, Field(default=None, description=description))
    title = "".join(part.capitalize() for part in (agent_name or "agent").split("_")) + "Output"
    return create_model(title, **fields)
