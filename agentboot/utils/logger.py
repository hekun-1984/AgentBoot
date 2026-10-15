"""统一日志。"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            continue


def setup_logging(level: str = "INFO") -> None:
    """配置 agentboot 日志树。重复调用只更新级别。"""
    global _CONFIGURED
    _ensure_utf8_stdio()
    logger = logging.getLogger("agentboot")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    logger_name = name if name.startswith("agentboot") else f"agentboot.{name}"
    return logging.getLogger(logger_name)
