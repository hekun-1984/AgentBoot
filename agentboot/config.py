"""配置管理。加载 config/settings.py、.env、环境变量和 config/models.yaml。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import yaml

from agentboot.exceptions import ConfigurationError

_TRUE = {"1", "true", "yes", "on"}
_ENGINES = {"native", "langgraph"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _load_project_module(path: Path) -> Any | None:
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("agentboot_project_settings", path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"Cannot load settings file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_models(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Model config must be a mapping: {path}")
    return loaded


class Settings:
    """运行时配置。显式环境变量优先于 .env，.env 优先于 config/settings.py。"""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        project = _load_project_module(self.project_root / "config" / "settings.py")
        file_env = _parse_env_file(self.project_root / ".env")
        env = dict(file_env)
        env.update(os.environ)
        models = _load_models(self.project_root / "config" / "models.yaml")
        default_profile = models.get("default") or {}
        if not isinstance(default_profile, dict):
            raise ConfigurationError("config/models.yaml default profile must be a mapping.")

        def pick(key: str, project_attr: str, fallback: Any) -> Any:
            if key in env and env[key] != "":
                return env[key]
            if project is not None and hasattr(project, project_attr):
                return getattr(project, project_attr)
            return fallback

        self.app_name = str(pick("AGENTBOOT_APP_NAME", "APP_NAME", "agentboot"))
        self.log_level = str(pick("AGENTBOOT_LOG_LEVEL", "LOG_LEVEL", "INFO")).upper()
        self.engine = str(pick("AGENTBOOT_ENGINE", "ENGINE", "native")).strip().lower()
        if self.engine not in _ENGINES:
            raise ConfigurationError(
                f"Unsupported AGENTBOOT_ENGINE '{self.engine}'. Choose native or langgraph."
            )
        self.demo = _as_bool(pick("AGENTBOOT_DEMO", "DEMO", True), default=True)
        self.llm_model = str(
            pick("AGENTBOOT_LLM_MODEL", "LLM_MODEL", default_profile.get("model", "gpt-4o-mini"))
        )
        self.llm_base_url = str(
            pick(
                "AGENTBOOT_LLM_BASE_URL",
                "LLM_BASE_URL",
                default_profile.get("base_url", "https://api.openai.com/v1"),
            )
        ).rstrip("/")
        self.llm_api_key = str(env.get("AGENTBOOT_LLM_API_KEY", "") or "")
        temperature = pick(
            "AGENTBOOT_LLM_TEMPERATURE",
            "LLM_TEMPERATURE",
            default_profile.get("temperature", 0),
        )
        self.llm_temperature = float(temperature)
        self.llm_timeout_seconds = float(default_profile.get("timeout_seconds", 60))
        self.llm_provider = str(default_profile.get("provider", "openai-compatible"))
        self.model_profiles = models.get("profiles") or {}
