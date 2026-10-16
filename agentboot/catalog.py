"""按用途管理模型。配置表放在这里，密钥从环境变量读取。"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentboot.exceptions import ConfigurationError
from agentboot.llm import LangChainModel, call_with_fallback

KNOWN_KINDS = {"chat", "embed", "rerank"}


@dataclass(frozen=True)
class ModelOption:
    name: str
    provider: str
    model: str
    api_key_env: str = ""
    timeout: float | None = None
    base_url: str = ""


@dataclass
class ModelSlot:
    name: str
    kind: str
    default: str
    options: list[ModelOption]

    def ordered(self) -> list[ModelOption]:
        primary = [item for item in self.options if item.name == self.default]
        backups = [item for item in self.options if item.name != self.default]
        return primary + backups


@dataclass
class EmbeddingClient:
    """向量化客户端。默认模型不可用时按名单换备选。"""

    candidates: list[tuple[str, Any]]
    last_option: str = ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        def call(model: Any) -> list[list[float]]:
            return [list(item) for item in model.embed_documents(list(texts))]

        name, vectors = call_with_fallback(self.candidates, call)
        self.last_option = name
        return vectors


@dataclass
class RerankClient:
    """精排客户端。返回 index 与 score，默认模型不可用时换备选。"""

    candidates: list[tuple[str, Any]]
    last_option: str = ""

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        def call(model: Any) -> list[dict]:
            raw = model.rerank(query, documents)
            return _normalize_rerank(raw)

        name, ranked = call_with_fallback(self.candidates, call)
        self.last_option = name
        return ranked


@dataclass
class ModelCatalog:
    """槽位目录。Agent 和主图只声明槽位名。"""

    slots: dict[str, ModelSlot] = field(default_factory=dict)
    _bound: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read(cls, path: str | Path) -> "ModelCatalog":
        file_path = Path(path)
        if not file_path.is_file():
            raise ConfigurationError(f"找不到模型配置: {file_path}")
        data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"{file_path} 的内容必须是映射")
        return cls.parse(data)

    @classmethod
    def parse(cls, data: dict) -> "ModelCatalog":
        if not isinstance(data, dict):
            raise ConfigurationError("模型配置必须是映射")
        slots: dict[str, ModelSlot] = {}
        for name, raw in data.items():
            slots[str(name)] = _parse_slot(str(name), raw)
        return cls(slots=slots)

    def bind(self, slot: str, client: Any) -> "ModelCatalog":
        """测试或自定义客户端直接占用一个槽位，不再按配置创建。"""
        self._bound[slot] = client
        return self

    def has(self, slot: str) -> bool:
        name = _requested(slot)
        return name in self.slots or name in self._bound

    def chat(self, slot: str) -> Any:
        name = _requested(slot)
        if name == "route" and name not in self.slots and name not in self._bound:
            name = "general"
        return self._resolve(name, "chat", build_chat, lambda pairs: LangChainModel(candidates=pairs))

    def embeddings(self, slot: str = "embed") -> EmbeddingClient:
        return self._resolve(slot or "embed", "embed", build_embedding, EmbeddingClient)

    def reranker(self, slot: str = "rerank") -> RerankClient:
        return self._resolve(slot or "rerank", "rerank", build_reranker, RerankClient)

    def _resolve(self, slot: str, kind: str, build: Any, wrap: Any) -> Any:
        name = _requested(slot)
        if name in self._bound:
            return self._bound[name]
        spec = self.slots.get(name)
        if spec is None:
            raise ConfigurationError(f"没有配置模型槽位 {name}")
        if spec.kind != kind:
            label = {"chat": "对话模型", "embed": "向量模型", "rerank": "精排模型"}
            raise ConfigurationError(f"槽位 {name} 的类型是 {label[spec.kind]}，不能当作 {label[kind]}")
        candidates = [(item.name, build(item)) for item in spec.ordered()]
        return wrap(candidates)


def resolve_api_key(option: ModelOption) -> str | None:
    if not option.api_key_env:
        return None
    value = os.getenv(option.api_key_env)
    if not value:
        raise ConfigurationError(f"环境变量 {option.api_key_env} 没有配置密钥")
    return value


def build_chat(option: ModelOption) -> Any:
    from langchain.chat_models import init_chat_model

    kwargs = _client_kwargs(option)
    if option.provider:
        return init_chat_model(option.model, model_provider=option.provider, **kwargs)
    return init_chat_model(option.model, **kwargs)


def build_embedding(option: ModelOption) -> Any:
    from langchain.embeddings import init_embeddings

    kwargs = _client_kwargs(option)
    if option.provider:
        return init_embeddings(option.model, provider=option.provider, **kwargs)
    return init_embeddings(option.model, **kwargs)


def build_reranker(option: ModelOption) -> Any:
    module_name = "langchain_" + option.provider.replace("-", "_")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        package = module_name.replace("_", "-")
        raise ConfigurationError(f"精排提供方 {option.provider} 需要安装 {package}") from exc
    factory = _rerank_class(module)
    if factory is None:
        raise ConfigurationError(f"{module_name} 里没有精排类")
    kwargs = _client_kwargs(option)
    kwargs["model"] = option.model
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise ConfigurationError(f"{option.provider} 的精排参数无法识别") from exc


def _client_kwargs(option: ModelOption) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    key = resolve_api_key(option)
    if key:
        kwargs["api_key"] = key
    if option.timeout is not None:
        kwargs["timeout"] = option.timeout
    if option.base_url:
        kwargs["base_url"] = option.base_url
    return kwargs


def _parse_slot(name: str, raw: object) -> ModelSlot:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"槽位 {name} 必须是映射")
    kind = str(raw.get("kind") or _default_kind(name)).strip()
    if kind not in KNOWN_KINDS:
        raise ConfigurationError(f"槽位 {name} 的类型无法识别: {kind}")
    options_raw = raw.get("options")
    if not isinstance(options_raw, list) or not options_raw:
        raise ConfigurationError(f"槽位 {name} 至少要有一个模型")
    options = [_parse_option(name, item) for item in options_raw]
    names = [item.name for item in options]
    if len(names) != len(set(names)):
        raise ConfigurationError(f"槽位 {name} 里有重名模型")
    default = str(raw.get("default") or "").strip()
    if not default:
        if len(options) == 1:
            default = options[0].name
        else:
            raise ConfigurationError(f"槽位 {name} 有多个模型，需要指定 default")
    if default not in names:
        raise ConfigurationError(f"槽位 {name} 的默认模型不在名单里: {default}")
    return ModelSlot(name=name, kind=kind, default=default, options=options)


def _parse_option(slot: str, raw: object) -> ModelOption:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"槽位 {slot} 的模型必须是映射")
    if "api_key" in raw:
        raise ConfigurationError(f"槽位 {slot} 的密钥请写入环境变量，配置里使用 api_key_env")
    name = str(raw.get("name") or "").strip()
    provider = str(raw.get("provider") or "").strip()
    model = str(raw.get("model") or "").strip()
    if not name or not provider or not model:
        raise ConfigurationError(f"槽位 {slot} 的每个模型都要有 name、provider 和 model")
    timeout = raw.get("timeout")
    if timeout is not None and not isinstance(timeout, (int, float)):
        raise ConfigurationError(f"槽位 {slot} 的 timeout 必须是数字")
    return ModelOption(
        name=name,
        provider=provider,
        model=model,
        api_key_env=str(raw.get("api_key_env") or "").strip(),
        timeout=float(timeout) if timeout is not None else None,
        base_url=str(raw.get("base_url") or "").strip(),
    )


def _default_kind(slot: str) -> str:
    if slot == "embed":
        return "embed"
    if slot == "rerank":
        return "rerank"
    return "chat"


def _requested(slot: str) -> str:
    text = (slot or "general").strip()
    return text or "general"


def _rerank_class(module: Any) -> Any:
    named = getattr(module, "Rerank", None)
    if named is not None:
        return named
    for attr in dir(module):
        if attr.endswith("Rerank") and attr[0].isupper():
            return getattr(module, attr)
    return None


def _normalize_rerank(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        raise TypeError("精排结果不是列表")
    ranked: list[dict] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            position = int(item.get("index", index))
            score = item.get("score", item.get("relevance_score", 0))
            ranked.append({"index": position, "score": float(score)})
            continue
        position = getattr(item, "index", index)
        score = getattr(item, "relevance_score", getattr(item, "score", 0))
        ranked.append({"index": int(position), "score": float(score)})
    return ranked
