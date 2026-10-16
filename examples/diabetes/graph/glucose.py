"""从用户原话里提取血糖数值。规则固定，不交给模型。"""

from __future__ import annotations

import re


_GLUCOSE_PATTERN = re.compile(r"血糖[^\d]{0,12}(\d+(?:\.\d+)?)")


def extract_glucose(text: str) -> float | None:
    """提取「血糖」附近的第一个数值，单位按 mmol/L 理解。"""
    matched = _GLUCOSE_PATTERN.search(text or "")
    if not matched:
        return None
    return float(matched.group(1))
