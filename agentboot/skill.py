"""从 SKILL.md 读取 Agent 契约。正文是系统提示词。"""

from __future__ import annotations

from pathlib import Path

import yaml

from agentboot.exceptions import SkillError
from agentboot.models import FAILURE_POLICIES, FieldSpec, SkillSpec


def _fields(raw: object) -> dict[str, FieldSpec]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise SkillError("input / output 必须是映射")
    fields: dict[str, FieldSpec] = {}
    for name, spec in raw.items():
        if isinstance(spec, str):
            fields[name] = FieldSpec(name=name, description=spec)
            continue
        if not isinstance(spec, dict):
            raise SkillError(f"字段 {name} 的声明无法识别")
        fields[name] = FieldSpec(
            name=name,
            type_name=str(spec.get("type", "str")),
            description=str(spec.get("description", "")),
            required=bool(spec.get("required", False)),
        )
    return fields


def load_skill(path: str | Path) -> SkillSpec:
    file_path = Path(path)
    if not file_path.is_file():
        raise SkillError(f"找不到 SKILL.md: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise SkillError(f"{file_path} 缺少 YAML front matter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError(f"{file_path} 的 front matter 没有闭合")
    data = yaml.safe_load(parts[1]) or {}
    if not isinstance(data, dict):
        raise SkillError(f"{file_path} 的 front matter 必须是映射")
    on_failure = str(data.get("on_failure", "return_partial"))
    if on_failure not in FAILURE_POLICIES:
        raise SkillError(f"{file_path} 的 on_failure 不合法: {on_failure}")
    tools = data.get("tools") or []
    if not isinstance(tools, list) or not all(isinstance(item, str) for item in tools):
        raise SkillError(f"{file_path} 的 tools 必须是字符串列表")
    return SkillSpec(
        name=str(data.get("name", "")).strip(),
        description=str(data.get("description", "")).strip(),
        tools=list(tools),
        max_steps=int(data.get("max_steps", 6)),
        on_failure=on_failure,
        inputs=_fields(data.get("input")),
        outputs=_fields(data.get("output")),
        system_prompt=parts[2].strip(),
    )
