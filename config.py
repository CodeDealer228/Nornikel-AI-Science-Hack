"""Centralized settings for the Nornikel Knowledge Graph.

All environment variables, paths, and tunables are collected here.
Modules that need configuration should import ``get_settings()``
(or accept a ``Settings`` instance for testability) rather than
reading ``os.environ`` directly.

Settings can be overridden at runtime via environment variables
(prefix ``NK_``) or by passing a ``Settings(...)`` instance to
module constructors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# Load a local .env file into os.environ before any settings dataclass reads
# it. No-op if python-dotenv is missing or no .env exists; safe to call at
# import. ``override=False`` keeps inline env vars winning over .env. Explicit
# path (repo-root .env) so it works regardless of the cwd the app is launched
# from. Must run before get_settings() is first called, hence module-level here
# rather than inside get_settings() (which is lru_cache'd).
try:  # pragma: no cover - environment-specific
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:  # pragma: no cover - dotenv absent
    pass


def _env(key: str, default: str | None = None) -> str | None:
    val = os.environ.get(key)
    return val if val not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687") or "bolt://localhost:7687")
    user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j") or "neo4j")
    password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "knowledge") or "knowledge")
    batch_size: int = field(default_factory=lambda: _env_int("NEO4J_BATCH_SIZE", 500))
    database: str = field(default_factory=lambda: _env("NEO4J_DATABASE", "neo4j") or "neo4j")


@dataclass(frozen=True)
class YandexGPTSettings:
    api_key: str = field(default_factory=lambda: _env("YANDEX_GPT_API_KEY", "") or "")
    folder_id: str = field(default_factory=lambda: _env("YANDEX_GPT_FOLDER_ID", "") or "")
    model_uri: str = field(default_factory=lambda: _env("YANDEX_GPT_MODEL_URI", "yandexgpt-lite") or "yandexgpt-lite")
    max_tokens: int = field(default_factory=lambda: _env_int("YANDEX_GPT_MAX_TOKENS", 3000))
    temperature: float = field(default_factory=lambda: _env_float("YANDEX_GPT_TEMPERATURE", 0.0))
    timeout: float = field(default_factory=lambda: _env_float("YANDEX_GPT_TIMEOUT", 60.0))
    max_retries: int = field(default_factory=lambda: _env_int("YANDEX_GPT_MAX_RETRIES", 3))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.folder_id)


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY", "") or "")
    model: str = field(default_factory=lambda: _env("OPENROUTER_MODEL", "poolside/laguna-xs-2.1:free") or "poolside/laguna-xs-2.1:free")
    base_url: str = field(default_factory=lambda: _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") or "https://openrouter.ai/api/v1")
    max_tokens: int = field(default_factory=lambda: _env_int("OPENROUTER_MAX_TOKENS", 3000))
    temperature: float = field(default_factory=lambda: _env_float("OPENROUTER_TEMPERATURE", 0.0))
    timeout: float = field(default_factory=lambda: _env_float("OPENROUTER_TIMEOUT", 60.0))
    max_retries: int = field(default_factory=lambda: _env_int("OPENROUTER_MAX_RETRIES", 3))
    http_referer: str = field(default_factory=lambda: _env("OPENROUTER_HTTP_REFERER", "") or "")
    app_title: str = field(default_factory=lambda: _env("OPENROUTER_APP_TITLE", "Nornikel-AI-Science-Hack") or "Nornikel-AI-Science-Hack")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class RouterSettings:
    graph_only_coverage: float = field(default_factory=lambda: _env_float("ROUTER_GRAPH_ONLY_COVERAGE", 0.62))
    graph_only_confidence: float = field(default_factory=lambda: _env_float("ROUTER_GRAPH_ONLY_CONFIDENCE", 0.55))
    graph_only_ambiguity: float = field(default_factory=lambda: _env_float("ROUTER_GRAPH_ONLY_AMBIGUITY", 0.70))
    hybrid_min_coverage: float = field(default_factory=lambda: _env_float("ROUTER_HYBRID_MIN_COVERAGE", 0.20))
    max_hops: int = field(default_factory=lambda: _env_int("ROUTER_MAX_HOPS", 4))
    max_paths: int = field(default_factory=lambda: _env_int("ROUTER_MAX_PATHS", 200))


@dataclass(frozen=True)
class IngestSettings:
    default_input_dir: Path = field(
        default_factory=lambda: Path(_env("INGEST_INPUT_DIR", "./Статьи") or "./Статьи")
    )
    default_output_dir: Path = field(
        default_factory=lambda: Path(_env("INGEST_OUTPUT_DIR", "./parsed_chunks") or "./parsed_chunks")
    )
    max_concurrency: int = field(default_factory=lambda: _env_int("INGEST_MAX_CONCURRENCY", 4))
    progress_every: int = field(default_factory=lambda: _env_int("INGEST_PROGRESS_EVERY", 10))


@dataclass(frozen=True)
class RAGSettings:
    backend: str = field(default_factory=lambda: _env("RAG_BACKEND", "in_memory") or "in_memory")
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 8))
    elasticsearch_url: str = field(default_factory=lambda: _env("RAG_ELASTICSEARCH_URL", "") or "")
    elasticsearch_index: str = field(default_factory=lambda: _env("RAG_ELASTICSEARCH_INDEX", "kg_chunks") or "kg_chunks")


@dataclass(frozen=True)
class APISettings:
    host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0") or "0.0.0.0")
    port: int = field(default_factory=lambda: _env_int("API_PORT", 8080))
    api_key: str = field(default_factory=lambda: _env("API_KEY", "") or "")
    enable_cors: bool = field(default_factory=lambda: _env_bool("API_ENABLE_CORS", True))
    request_timeout_sec: int = field(default_factory=lambda: _env_int("API_REQUEST_TIMEOUT", 120))


@dataclass(frozen=True)
class LoggingSettings:
    level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO") or "INFO")
    format: str = field(default_factory=lambda: _env("LOG_FORMAT", "text") or "text")
    log_dir: Path | None = field(
        default_factory=lambda: Path(_env("LOG_DIR", "")) if _env("LOG_DIR") else None
    )


@dataclass(frozen=True)
class Settings:
    """Top-level settings container."""
    neo4j: Neo4jSettings = field(default_factory=Neo4jSettings)
    yandex_gpt: YandexGPTSettings = field(default_factory=YandexGPTSettings)
    openrouter: OpenRouterSettings = field(default_factory=OpenRouterSettings)
    router: RouterSettings = field(default_factory=RouterSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    api: APISettings = field(default_factory=APISettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance.

    Tests should call ``get_settings.cache_clear()`` after mutating
    env vars or construct a ``Settings(...)`` directly.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Reset the lru_cache (mainly for tests)."""
    get_settings.cache_clear()


__all__ = [
    "APISettings",
    "IngestSettings",
    "LoggingSettings",
    "Neo4jSettings",
    "OpenRouterSettings",
    "RAGSettings",
    "RouterSettings",
    "Settings",
    "YandexGPTSettings",
    "get_settings",
    "reset_settings_cache",
]
