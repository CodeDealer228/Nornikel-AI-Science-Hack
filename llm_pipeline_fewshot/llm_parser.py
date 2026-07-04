"""
LLM extraction pipeline for the Knowledge Graph ingestion layer.

Talks to Yandex Foundation Models (YandexGPT / YandexGPT-Lite) over
the legacy completion endpoint, drives a chunk through the
NER+RE prompt in ``ner_re_extraction_prompt.md``, parses the JSON
response into ``LLMExtraction`` and wraps each chunk attempt in
``ChunkExtractionResult`` for downstream loading.

The HTTP layer is intentionally minimal — just ``urllib`` from the
standard library — to avoid pinning a third-party HTTP client. The
caller is responsible for batching and concurrency.

Usage::

    client = YandexGPTClient(
        api_key=os.environ["YANDEX_GPT_API_KEY"],
        folder_id=os.environ["YANDEX_GPT_FOLDER_ID"],
        model_uri="yandexgpt-lite",  # or "yandexgpt", "qwen2-72b-instruct"
    )
    extractor = ChunkExtractor(client, prompt_loader=load_prompt)
    result = await extractor.extract_chunk(chunk)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

from .models import (
    ChunkExtractionResult,
    ChunkInput,
    ChunkProvenance,
    EnrichedEntity,
    EnrichedRelation,
    LLMEntity,
    LLMExtraction,
    LLMRelation,
    RelationType,
    ValidationIssue,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "yandexgpt-lite"
COMPLETION_ENDPOINT = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "poolside/laguna-xs-2.1:free"
DEFAULT_MAX_TOKENS = 3000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 3
RETRY_BACKOFF = 1.6

_PROMPT_LOADERS: dict[str, Callable[[], str]] = {}


def register_prompt_loader(name: str, loader: Callable[[], str]) -> None:
    """Register a system-prompt loader by name."""
    _PROMPT_LOADERS[name] = loader


def load_default_prompt() -> str:
    """Load the bundled NER/RE system prompt."""
    from pathlib import Path

    prompt_path = Path(__file__).with_name("ner_re_extraction_prompt.md")
    return prompt_path.read_text(encoding="utf-8")


register_prompt_loader("ner_re", load_default_prompt)


# ---------------------------------------------------------------------------
# Low-level HTTP client
# ---------------------------------------------------------------------------


@dataclass
class CompletionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CompletionResponse:
    text: str
    usage: CompletionUsage
    model_version: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class YandexGPTError(RuntimeError):
    """Raised when the configured LLM provider returns an error.

    DeepSeek hosted on Yandex Cloud surfaces through the same endpoint and
    auth, so its failures raise ``YandexGPTError`` too — no separate
    DeepSeek-specific error type. OpenRouter failures also use this base
    error so the ingestion fallback path stays provider-agnostic.
    """


class LLMClient(Protocol):
    """Минимальный интерфейс LLM-клиента для извлечения NER+RE.

    Реализация может быть реальной (YandexGPT) или детерминированной mock-
    версией для smoke-test без токенов и сетевых вызовов.
    """

    model_uri: str

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResponse:
        """Вернуть завершение модели в едином формате."""
        ...

    async def acomplete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResponse:
        """Асинхронная обёртка над ``complete``."""
        ...


class YandexGPTClient:
    """Minimal async-compatible client for Yandex Foundation Models.

    Uses ``urllib`` from the standard library to avoid pinning a
    third-party HTTP client. Supports API-key auth (header) and
    the legacy ``/completion`` endpoint, which is the most stable
    across YandexGPT generations.

    The same ``/completion`` endpoint and ``Api-Key`` auth serve every
    Foundation Model hosted on Yandex Cloud / Yandex AI Studio — including
    DeepSeek, whose model URI uses a ``ds://`` scheme. Pass a full
    ``ds://<folder>/deepseek...`` URI as ``model_uri`` (or via
    ``YANDEX_GPT_MODEL_URI``) and the same client works for DeepSeek without
    a separate class; auth is the regular ``YANDEX_GPT_API_KEY``, not a
    DeepSeek-issued key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        folder_id: str | None = None,
        model_uri: str | None = None,
        endpoint: str = COMPLETION_ENDPOINT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.api_key = api_key or os.environ.get("YANDEX_GPT_API_KEY", "")
        self.folder_id = folder_id or os.environ.get("YANDEX_GPT_FOLDER_ID", "")
        self.model_uri = model_uri or os.environ.get(
            "YANDEX_GPT_MODEL_URI", DEFAULT_MODEL
        )
        if not self.api_key:
            raise YandexGPTError(
                "YandexGPT API key is required (set YANDEX_GPT_API_KEY)."
            )
        if "://" in self.model_uri:
            # A pre-resolved URI with a scheme (gpt://, ds://, etc.) is used
            # verbatim — this is how DeepSeek-on-Yandex (ds://<folder>/...)
            # and any folder-qualified YandexGPT URI are passed in.
            pass
        elif self.folder_id:
            self.model_uri = f"gpt://{self.folder_id}/{self.model_uri}"
        else:
            raise YandexGPTError(
                "model_uri must be a full URI (e.g. 'gpt://<folder>/<model>' "
                "or 'ds://<folder>/deepseek...') or a bare model name with "
                "YANDEX_GPT_FOLDER_ID set."
            )

        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max(1, max_retries)

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": (
                    self.temperature if temperature is None else temperature
                ),
                "maxTokens": str(max_tokens or self.max_tokens),
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise YandexGPTError(
                f"YandexGPT HTTP {exc.code} on {self.endpoint}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise YandexGPTError(
                f"YandexGPT network error on {self.endpoint}: {exc.reason}"
            ) from exc

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResponse:
        """Run a single completion with bounded retries on transient errors."""
        payload = self._build_payload(system_prompt, user_prompt, max_tokens, temperature)
        last_exc: Exception | None = None
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._post(payload)
                return self._parse_response(response)
            except YandexGPTError as exc:
                last_exc = exc
                # Retry on 429/5xx, surface others immediately.
                if "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
                    raise
                if attempt < self.max_retries:
                    log.warning(
                        "YandexGPT attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= RETRY_BACKOFF
        raise YandexGPTError(
            f"YandexGPT failed after {self.max_retries} attempts: {last_exc}"
        )

    async def acomplete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResponse:
        """Async wrapper — ``complete()`` is already non-blocking (urllib is sync,
        but called via ``run_in_executor`` semantics in the orchestrator)."""
        return self.complete(system_prompt, user_prompt, max_tokens, temperature)

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> CompletionResponse:
        try:
            result = payload["result"]
            alternatives = result["alternatives"]
            text = alternatives[0]["message"]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise YandexGPTError(
                f"Malformed YandexGPT response: {payload!r}"
            ) from exc

        usage_payload = result.get("usage", {}) or {}
        usage = CompletionUsage(
            input_tokens=_safe_int(usage_payload.get("inputTextTokens")),
            output_tokens=_safe_int(usage_payload.get("completionTokens")),
            total_tokens=_safe_int(usage_payload.get("totalTokens")),
        )
        return CompletionResponse(
            text=text,
            usage=usage,
            model_version=result.get("modelVersion", ""),
            raw=payload,
        )


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """Minimal OpenRouter client using the chat completions API.

    The client intentionally mirrors ``YandexGPTClient`` and uses only
    ``urllib``. It implements the same ``LLMClient`` protocol, so ingestion can
    switch providers through ``LLM_CLIENT_MODE=openrouter`` without touching
    chunking, ensemble, or graph loading code.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        http_referer: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise YandexGPTError(
                "OpenRouter API key is required (set OPENROUTER_API_KEY)."
            )
        self.model_uri = model or os.environ.get(
            "OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL
        )
        root = (base_url or os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
        self.endpoint = f"{root}/chat/completions"
        self.max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS") or max_tokens)
        self.temperature = float(os.environ.get("OPENROUTER_TEMPERATURE") or temperature)
        self.timeout = float(os.environ.get("OPENROUTER_TIMEOUT") or timeout)
        self.max_retries = max(
            1,
            int(os.environ.get("OPENROUTER_MAX_RETRIES") or max_retries),
        )
        self.http_referer = http_referer if http_referer is not None else os.environ.get("OPENROUTER_HTTP_REFERER", "")
        self.app_title = app_title if app_title is not None else os.environ.get(
            "OPENROUTER_APP_TITLE", "Nornikel-AI-Science-Hack"
        )

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return {
            "model": self.model_uri,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(max_tokens or self.max_tokens),
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise YandexGPTError(
                f"OpenRouter HTTP {exc.code} on {self.endpoint}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise YandexGPTError(
                f"OpenRouter network error on {self.endpoint}: {exc.reason}"
            ) from exc

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResponse:
        payload = self._build_payload(system_prompt, user_prompt, max_tokens, temperature)
        last_exc: Exception | None = None
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._post(payload)
                return self._parse_response(response)
            except YandexGPTError as exc:
                last_exc = exc
                if "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
                    raise
                if attempt < self.max_retries:
                    log.warning(
                        "OpenRouter attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= RETRY_BACKOFF
        raise YandexGPTError(
            f"OpenRouter failed after {self.max_retries} attempts: {last_exc}"
        )

    async def acomplete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResponse:
        return self.complete(system_prompt, user_prompt, max_tokens, temperature)

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> CompletionResponse:
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise YandexGPTError(
                f"Malformed OpenRouter response: {payload!r}"
            ) from exc
        if isinstance(content, list):
            text = "".join(
                str(part.get("text") or part.get("content") or "")
                if isinstance(part, dict)
                else str(part)
                for part in content
            )
        else:
            text = str(content)

        usage_payload = payload.get("usage", {}) or {}
        usage = CompletionUsage(
            input_tokens=_safe_int(usage_payload.get("prompt_tokens")),
            output_tokens=_safe_int(usage_payload.get("completion_tokens")),
            total_tokens=_safe_int(usage_payload.get("total_tokens")),
        )
        return CompletionResponse(
            text=text,
            usage=usage,
            model_version=str(payload.get("model") or ""),
            raw=payload,
        )


# ---------------------------------------------------------------------------
# DeepSeek (hosted on Yandex Cloud / Yandex AI Studio)
# ---------------------------------------------------------------------------
#
# DeepSeek is NOT called via the public api.deepseek.com endpoint here. In this
# project it is hosted on Yandex Foundation Models, so it reuses the same
# ``/completion`` endpoint, ``Api-Key`` auth and ``{modelUri, completionOptions,
# messages}`` request shape as ``YandexGPTClient`` — only the model URI changes
# to a ``ds://`` scheme (e.g. ``ds://<folder_id>/deepseek-v3`` etc.). Select it
# through ``create_llm_client("deepseek")`` or ``LLM_CLIENT_MODE=deepseek``;
# auth is the regular ``YANDEX_GPT_API_KEY``, not a DeepSeek-issued key.


# ---------------------------------------------------------------------------
# Mock client and provider switch
# ---------------------------------------------------------------------------

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on", "mock"}
_MOCK_MODEL_URI = "mock://ner-re-examples"


class MockLLMClient:
    """Детерминированный LLM-клиент для smoke-test без токенов и сети.

    Клиент возвращает JSON в той же схеме, что и реальный NER+RE промпт. Фикстуры
    составлены по ручным примерам из ``ner_re_extraction/ner_re_examples.md`` и
    небольшому fallback-набору общеупотребимых терминов, чтобы end-to-end прогон
    мог проверить связку chunking → llm_parser → ensemble без доступа к API.
    """

    def __init__(
        self,
        model_uri: str = _MOCK_MODEL_URI,
        examples_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.model_uri = model_uri
        self.examples_path = os.fspath(examples_path) if examples_path else ""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResponse:
        """Вернуть заранее подготовленное извлечение для текста чанка."""
        del system_prompt, max_tokens, temperature
        payload = self._build_payload(user_prompt)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return CompletionResponse(
            text=text,
            usage=CompletionUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            model_version="mock-v1",
            raw={"provider": "mock", "examples_path": self.examples_path},
        )

    async def acomplete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResponse:
        """Асинхронная обёртка над mock-ответом."""
        return self.complete(system_prompt, user_prompt, max_tokens, temperature)

    def _build_payload(self, user_prompt: str) -> dict[str, list[dict[str, Any]]]:
        chunk_text = _extract_chunk_text_from_prompt(user_prompt)
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []

        if self._looks_like_lead_sorption(chunk_text):
            entities, relations = self._lead_sorption_fixture(chunk_text)
        elif self._looks_like_vanyukov_furnace(chunk_text):
            entities, relations = self._vanyukov_fixture(chunk_text)
        else:
            entities, relations = self._keyword_fallback(chunk_text)

        return {"entities": entities, "relations": relations}

    @staticmethod
    def _looks_like_lead_sorption(text: str) -> bool:
        lower = text.lower()
        return any(token in lower for token in ("lewatit", "свин", "сорбц"))

    @staticmethod
    def _looks_like_vanyukov_furnace(text: str) -> bool:
        lower = text.lower()
        return "ванюков" in lower or "пвк" in lower or "медный штейн" in lower

    def _lead_sorption_fixture(
        self,
        text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []

        process = _make_mock_entity(
            "e1",
            "Process",
            "очистка сорбционная от свинца",
            text,
            ("очистка сорбционная от свинца", "сорбционная очистка", "сорбция свинца", "десорбция свинца"),
        )
        material = _make_mock_entity(
            "e2",
            "Material",
            "анионит Lewatit А365",
            text,
            ("анионит Lewatit А365", "Lewatit А365", "Lewatit A365"),
        )
        condition = _make_mock_entity(
            "e3",
            "Condition",
            "pH 3,0–3,5",
            text,
            ("pH 3,0–3,5", "pH 3,0-3,5", "pH 3,0", "pH"),
            attributes={"value_raw": "pH 3,0–3,5"},
        )

        for entity in (process, material, condition):
            if entity is not None:
                entities.append(entity)

        ids = {entity["local_id"] for entity in entities}
        if {"e1", "e2"} <= ids:
            relations.append({
                "subject": "e1",
                "predicate": "uses_material",
                "object": "e2",
                "quote": _short_quote(text, (process, material)),
            })
        if {"e1", "e3"} <= ids:
            relations.append({
                "subject": "e1",
                "predicate": "operates_at_condition",
                "object": "e3",
                "quote": _short_quote(text, (process, condition)),
            })
        return entities, relations

    def _vanyukov_fixture(
        self,
        text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []

        equipment = _make_mock_entity(
            "e1",
            "Equipment",
            "печь Ванюкова конвертерная (ПВК)",
            text,
            ("печь Ванюкова конвертерная", "ПВК", "печь Ванюкова"),
        )
        material = _make_mock_entity(
            "e2",
            "Material",
            "медный штейн",
            text,
            ("медный штейн", "медных штейнов", "штейн"),
        )
        output = _make_mock_entity(
            "e3",
            "Material",
            "черновая медь",
            text,
            ("черновая медь", "черновой меди", "медь"),
        )

        for entity in (equipment, material, output):
            if entity is not None:
                entities.append(entity)

        ids = {entity["local_id"] for entity in entities}
        if {"e1", "e2"} <= ids:
            relations.append({
                "subject": "e1",
                "predicate": "uses_material",
                "object": "e2",
                "quote": _short_quote(text, (equipment, material)),
            })
        if {"e1", "e3"} <= ids:
            relations.append({
                "subject": "e1",
                "predicate": "produces_output",
                "object": "e3",
                "quote": _short_quote(text, (equipment, output)),
            })
        return entities, relations

    def _keyword_fallback(
        self,
        text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        specs: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
            ("e1", "Material", "никель", ("никель", "никеля", "никелевый")),
            ("e2", "Material", "медь", ("медь", "меди", "медный")),
            ("e3", "Process", "флотация", ("флотация", "флотации")),
            ("e4", "Property", "температура", ("температура", "°C")),
            ("e5", "Condition", "pH", ("pH", "рН")),
        )
        entities: list[dict[str, Any]] = []
        for local_id, entity_type, canonical_name, candidates in specs:
            entity = _make_mock_entity(local_id, entity_type, canonical_name, text, candidates)
            if entity is not None:
                entities.append(entity)
        return entities, []


def _extract_chunk_text_from_prompt(user_prompt: str) -> str:
    """Достать исходный текст чанка из user prompt, созданного ``_build_user_prompt``."""
    lines = user_prompt.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# ID чанка:"):
            return "\n".join(lines[index + 2:]).strip()
    return user_prompt.strip()


def _find_first_mention(text: str, candidates: Sequence[str]) -> str | None:
    """Найти первую форму сущности, реально встречающуюся в чанке."""
    lower = text.lower()
    for candidate in candidates:
        if candidate and candidate.lower() in lower:
            start = lower.index(candidate.lower())
            return text[start:start + len(candidate)]
    return None


def _make_mock_entity(
    local_id: str,
    entity_type: str,
    canonical_name: str,
    text: str,
    mention_candidates: Sequence[str],
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Собрать сущность mock-ответа только если mention найден в тексте."""
    mention = _find_first_mention(text, mention_candidates)
    if mention is None:
        return None
    payload: dict[str, Any] = {
        "local_id": local_id,
        "type": entity_type,
        "canonical_name": canonical_name,
        "mentions": [mention],
        "attributes": attributes or {},
    }
    return payload


def _short_quote(
    text: str,
    entities: Sequence[dict[str, Any] | None],
    max_chars: int = 240,
) -> str:
    """Вернуть короткую цитату вокруг первого найденного mention."""
    mentions = [
        entity["mentions"][0]
        for entity in entities
        if entity and entity.get("mentions")
    ]
    lower = text.lower()
    positions = [lower.find(mention.lower()) for mention in mentions if mention]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return text[:max_chars].strip()
    start = max(0, min(positions) - 80)
    end = min(len(text), start + max_chars)
    return text[start:end].strip()


def _env_requests_mock() -> bool:
    """Проверить env-переключатели mock-режима."""
    explicit = os.environ.get("YANDEX_GPT_USE_MOCK", "").strip().lower()
    if explicit in _TRUTHY_ENV_VALUES:
        return True
    return False


def create_llm_client(mode: str | None = None) -> LLMClient:
    """Создать LLM-клиент по режиму ``mock``/``real``/``deepseek``/``openrouter``.

    Режим берётся из аргумента или env-переменных ``LLM_CLIENT_MODE`` / ``LLM_MODE``.
    Дополнительно ``YANDEX_GPT_USE_MOCK=1`` принудительно включает mock. По
    умолчанию используется реальный YandexGPT-клиент, чтобы не менять поведение
    существующего продового кода.

    ``deepseek`` — DeepSeek, хостится на Yandex Cloud / Yandex AI Studio: тот же
    ``/completion`` endpoint и ``Api-Key`` auth, что и YandexGPT, только
    ``modelUri`` в схеме ``ds://``. Аутентификация — обычным
    ``YANDEX_GPT_API_KEY`` (не DeepSeek-ключом). URI модели разрешается так:
    если ``YANDEX_GPT_MODEL_URI`` уже задаёт ``ds://``-URI — он используется
    as-is; иначе собирается ``ds://<YANDEX_GPT_FOLDER_ID>/<DEEPSEEK_MODEL>``
    (``DEEPSEEK_MODEL`` по умолчанию ``deepseek-v3``).

    ``openrouter`` — OpenRouter chat completions API. Нужен
    ``OPENROUTER_API_KEY``; модель задаётся через ``OPENROUTER_MODEL`` и по
    умолчанию указывает на бесплатный ``*:free`` slug.
    """
    selected = (
        mode
        or os.environ.get("LLM_CLIENT_MODE")
        or os.environ.get("LLM_MODE")
        or ("mock" if _env_requests_mock() else "real")
    )
    normalized = selected.strip().lower()
    if normalized in {"mock", "offline", "test", "fake"}:
        examples_path = os.environ.get("LLM_MOCK_EXAMPLES_PATH")
        return MockLLMClient(examples_path=examples_path)
    if normalized in {"real", "yandex", "yandexgpt", "yandexgpt-lite"}:
        return YandexGPTClient()
    if normalized in {"deepseek", "deepseek-chat", "deepseek-reasoner", "deepseek-v3"}:
        return _build_deepseek_via_yandex()
    if normalized in {"openrouter", "openrouter-free", "router"}:
        return OpenRouterClient()
    raise YandexGPTError(
        "Unsupported LLM client mode "
        f"'{selected}'. Use 'mock', 'real'/'yandex', 'deepseek' or 'openrouter'."
    )


def _build_deepseek_via_yandex() -> YandexGPTClient:
    """DeepSeek-on-Yandex: a ``YandexGPTClient`` with a ``ds://`` model URI.

    Auth is ``YANDEX_GPT_API_KEY``. Model URI resolution:
      * ``YANDEX_GPT_MODEL_URI`` starting with ``ds://`` → used verbatim;
      * otherwise ``ds://<YANDEX_GPT_FOLDER_ID>/<DEEPSEEK_MODEL>``
        (``DEEPSEEK_MODEL`` default ``deepseek-v3``), requiring folder_id.
    """
    env_uri = os.environ.get("YANDEX_GPT_MODEL_URI", "")
    if env_uri.startswith("ds://"):
        return YandexGPTClient(model_uri=env_uri)
    folder_id = os.environ.get("YANDEX_GPT_FOLDER_ID", "")
    if not folder_id:
        raise YandexGPTError(
            "DeepSeek-on-Yandex needs either a ds:// YANDEX_GPT_MODEL_URI or "
            "YANDEX_GPT_FOLDER_ID set (to build ds://<folder>/<model>)."
        )
    deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v3")
    model_uri = f"ds://{folder_id}/{deepseek_model}"
    return YandexGPTClient(model_uri=model_uri)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"`{3}(?:json)?\s*(\{.*?\})\s*`{3}", re.S | re.I)


def extract_json_object(text: str) -> str:
    """Pull a single JSON object out of a free-form LLM response.

    Handles fenced code blocks (``\\`\\`\\`json { ... } \\`\\`\\```) and
    recovers the first balanced top-level object if the response is
    prose-wrapped.
    """
    stripped = text.strip()
    block = _JSON_BLOCK_RE.search(stripped)
    if block:
        stripped = block.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start == -1:
        raise ValueError("response does not contain a JSON object")

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start:i + 1]
    raise ValueError("response contains an unfinished JSON object")


def parse_llm_json(text: str) -> LLMExtraction:
    """Parse the raw LLM text into a validated ``LLMExtraction``."""
    obj = json.loads(extract_json_object(text))
    if hasattr(LLMExtraction, "model_validate"):
        return LLMExtraction.model_validate(obj)
    # Fallback path: the lightweight ``BaseModel`` shim used when pydantic
    # is not installed. Coerce nested dicts into their model classes so the
    # rest of the pipeline can rely on attribute access.
    entities_raw = obj.get("entities", []) or []
    relations_raw = obj.get("relations", []) or []
    entities = [
        LLMEntity(**entity) if isinstance(entity, dict) else entity
        for entity in entities_raw
    ]
    relations = [
        LLMRelation(**relation) if isinstance(relation, dict) else relation
        for relation in relations_raw
    ]
    return LLMExtraction(entities=entities, relations=relations)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_UNIT_HINT_RE = re.compile(
    r"(?:°C|г/л|мг/л|мг/дм3|м3/ч|м3/сут|т/сут|кг/ч|кВт|МВт|мм/сут|"
    r"pH|атм|МПа|Па|об%|ppm|%|°)",
    re.IGNORECASE | re.UNICODE,
)
_NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)?", re.UNICODE)


def _validate_entity(
    raw: dict[str, Any],
    text: str,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    cleaned = dict(raw)
    local_id = str(cleaned.get("local_id") or "").strip()
    if not local_id:
        issues.append(ValidationIssue(severity="error", code="missing_local_id"))
    cleaned["local_id"] = local_id

    mentions = [
        str(m).strip() for m in (cleaned.get("mentions") or []) if str(m).strip()
    ]
    cleaned["mentions"] = mentions

    for mention in mentions:
        if mention and mention not in text:
            issues.append(ValidationIssue(
                severity="warning",
                code="mention_not_in_text",
                local_id=local_id,
                message=f"mention '{mention[:60]}' is not a substring of the chunk text",
            ))

    if cleaned.get("type") == "Property":
        attrs = cleaned.get("attributes") or {}
        value_raw = str(attrs.get("value_raw") or "").strip()
        if value_raw:
            if not _UNIT_HINT_RE.search(value_raw) and not attrs.get("unit"):
                issues.append(ValidationIssue(
                    severity="warning",
                    code="property_value_no_unit",
                    local_id=local_id,
                    message="Property value is missing a unit hint.",
                ))
            for number in _NUMERIC_RE.findall(value_raw):
                token = number.replace(",", ".")
                if token and token not in value_raw.replace(",", "."):
                    issues.append(ValidationIssue(
                        severity="warning",
                        code="numeric_token_mismatch",
                        local_id=local_id,
                    ))
                    break

    return cleaned, issues


def _validate_relations(
    raw_relations: Iterable[dict[str, Any]],
    entity_ids: set[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    valid_predicates = {str(p) for p in RelationType}
    for index, raw in enumerate(raw_relations):
        predicate = str(raw.get("predicate") or "").strip()
        if predicate not in valid_predicates:
            issues.append(ValidationIssue(
                severity="warning",
                code="unknown_predicate",
                relation_index=index,
                message=f"predicate '{predicate}' is not in the ontology",
            ))
        subject = str(raw.get("subject") or "").strip()
        obj = str(raw.get("object") or "").strip()
        if subject and subject not in entity_ids:
            issues.append(ValidationIssue(
                severity="error",
                code="relation_endpoint_missing",
                relation_index=index,
                message=f"relation subject '{subject}' is not in the entity set",
            ))
        if obj and obj not in entity_ids:
            issues.append(ValidationIssue(
                severity="error",
                code="relation_endpoint_missing",
                relation_index=index,
                message=f"relation object '{obj}' is not in the entity set",
            ))
    return issues


# ---------------------------------------------------------------------------
# ChunkExtractor
# ---------------------------------------------------------------------------

_DEFAULT_ENTITY_CONFIDENCE = 0.85
_DEFAULT_RELATION_CONFIDENCE = 0.78


@dataclass
class _ProvenanceContext:
    chunk_id: str
    source_document: str
    char_start: int
    char_end: int
    page: int | None = None


def _build_user_prompt(chunk: ChunkInput) -> str:
    parts: list[str] = []
    provenance = chunk.provenance
    if provenance.heading_path:
        parts.append(f"# Заголовки: {' / '.join(provenance.heading_path)}")
    parts.append(f"# ID чанка: {chunk.chunk_id}")
    parts.append("")
    parts.append(chunk.text.strip())
    return "\n".join(parts)


class ChunkExtractor:
    """Run the NER+RE prompt on a single chunk and return a ``ChunkExtractionResult``."""

    def __init__(
        self,
        client: LLMClient,
        prompt_name: str = "ner_re",
        default_entity_confidence: float = _DEFAULT_ENTITY_CONFIDENCE,
        default_relation_confidence: float = _DEFAULT_RELATION_CONFIDENCE,
    ) -> None:
        self._client = client
        self._prompt_name = prompt_name
        self._default_entity_confidence = default_entity_confidence
        self._default_relation_confidence = default_relation_confidence
        try:
            self._system_prompt = _PROMPT_LOADERS[prompt_name]()
        except KeyError as exc:
            raise YandexGPTError(
                f"prompt loader '{prompt_name}' is not registered"
            ) from exc

    def extract_chunk(self, chunk: ChunkInput) -> ChunkExtractionResult:
        provenance = chunk.provenance
        result = ChunkExtractionResult(
            chunk_id=chunk.chunk_id,
            status="ok",
            source_document=provenance.source_document,
            char_start=provenance.char_start,
            char_end=provenance.char_end,
            page=provenance.page,
        )
        if not chunk.text.strip():
            result.status = "empty"
            return result

        user_prompt = _build_user_prompt(chunk)
        attempts = 0
        try:
            response = self._client.complete(self._system_prompt, user_prompt)
        except YandexGPTError as exc:
            attempts += 1
            result.status = "error"
            result.error = str(exc)
            result.attempts = attempts
            return result

        attempts += 1
        result.attempts = attempts
        result.model_uri = self._client.model_uri
        result.raw_response = response.text
        result.usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
            "model_version": response.model_version,
        }

        try:
            extraction = parse_llm_json(response.text)
        except (ValueError, json.JSONDecodeError) as exc:
            result.status = "error"
            result.error = f"json_parse_error: {exc}"
            return result

        result.entities, result.relations, issues = self._materialise(chunk, extraction)
        result.issues = issues
        return result

    def _materialise(
        self,
        chunk: ChunkInput,
        extraction: LLMExtraction,
    ) -> tuple[list[EnrichedEntity], list[EnrichedRelation], list[ValidationIssue]]:
        issues: list[ValidationIssue] = []
        entities: list[EnrichedEntity] = []
        valid_local_ids: set[str] = set()
        provenance = chunk.provenance
        text = chunk.text
        text_normalised = " ".join(text.split())

        for raw_entity in extraction.entities:
            raw_dict = raw_entity.model_dump() if hasattr(raw_entity, "model_dump") else dict(raw_entity)
            cleaned, entity_issues = _validate_entity(raw_dict, text_normalised)
            issues.extend(entity_issues)
            if not cleaned.get("local_id"):
                continue
            try:
                entity = EnrichedEntity(
                    entity=str(cleaned.get("canonical_name") or "").strip()
                    or (cleaned["mentions"][0] if cleaned.get("mentions") else ""),
                    type=cleaned["type"],
                    chunk_id=chunk.chunk_id,
                    source_document=provenance.source_document,
                    page=provenance.page,
                    quote=cleaned["mentions"][0] if cleaned.get("mentions") else "",
                    confidence=self._default_entity_confidence,
                    local_id=cleaned["local_id"],
                    mentions=list(cleaned.get("mentions") or []),
                    attributes=cleaned.get("attributes") or {},
                    heading_path=list(provenance.heading_path or []),
                    char_start=provenance.char_start,
                    char_end=provenance.char_end,
                    extractor="yandex_gpt",
                    needs_review=any(
                        issue.code in {"mention_not_in_text", "property_value_no_unit"}
                        for issue in entity_issues
                    ),
                )
            except Exception as exc:  # pragma: no cover - defensive
                issues.append(ValidationIssue(
                    severity="error",
                    code="entity_construction_error",
                    local_id=cleaned.get("local_id"),
                    message=str(exc),
                ))
                continue
            entities.append(entity)
            valid_local_ids.add(entity.local_id)

        relation_issues = _validate_relations(
            (rel.model_dump() if hasattr(rel, "model_dump") else dict(rel)
             for rel in extraction.relations),
            valid_local_ids,
        )
        issues.extend(relation_issues)

        relations: list[EnrichedRelation] = []
        entity_by_id = {entity.local_id: entity for entity in entities}
        for raw_relation in extraction.relations:
            raw_dict = raw_relation.model_dump() if hasattr(raw_relation, "model_dump") else dict(raw_relation)
            subject_id = str(raw_dict.get("subject") or "").strip()
            object_id = str(raw_dict.get("object") or "").strip()
            if subject_id not in entity_by_id or object_id not in entity_by_id:
                continue
            try:
                subject = entity_by_id[subject_id]
                obj = entity_by_id[object_id]
                relation = EnrichedRelation(
                    source_entity=subject.entity,
                    target_entity=obj.entity,
                    relation_type=raw_dict["predicate"],
                    chunk_id=chunk.chunk_id,
                    source_document=provenance.source_document,
                    page=provenance.page,
                    quote=str(raw_dict.get("quote") or ""),
                    confidence=self._default_relation_confidence,
                    source_entity_type=subject.type,
                    target_entity_type=obj.type,
                    source_local_id=subject.local_id,
                    target_local_id=obj.local_id,
                    heading_path=list(provenance.heading_path or []),
                    char_start=provenance.char_start,
                    char_end=provenance.char_end,
                    extractor="yandex_gpt",
                    needs_review=False,
                    note=(
                        str(raw_dict.get("note"))
                        if raw_dict.get("note") and str(raw_dict.get("predicate")) == "contradicts"
                        else None
                    ),
                )
            except Exception as exc:  # pragma: no cover - defensive
                issues.append(ValidationIssue(
                    severity="error",
                    code="relation_construction_error",
                    message=str(exc),
                ))
                continue
            relations.append(relation)

        return entities, relations, issues


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------


class ChunkBatchRunner:
    """Run ``ChunkExtractor.extract_chunk`` over a sequence of chunks with bounded
    concurrency. Designed to be awaited from an ingestion orchestrator.
    """

    def __init__(
        self,
        client: LLMClient,
        prompt_name: str = "ner_re",
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._extractor = ChunkExtractor(client=client, prompt_name=prompt_name)
        self._semaphore: Any = None  # lazily created per-loop

    async def run(self, chunks: Sequence[ChunkInput]) -> list[ChunkExtractionResult]:
        import asyncio

        semaphore = asyncio.Semaphore(len(chunks) and 4 or 1)
        if hasattr(self, "_max_concurrency"):
            semaphore = asyncio.Semaphore(getattr(self, "_max_concurrency"))

        async def _one(chunk: ChunkInput) -> ChunkExtractionResult:
            async with semaphore:
                return await asyncio.to_thread(self._extractor.extract_chunk, chunk)

        return await asyncio.gather(*(_one(c) for c in chunks))


# ---------------------------------------------------------------------------
# Convenience: chunk → ChunkInput conversion for tests / scripts
# ---------------------------------------------------------------------------


def make_chunk_input(
    chunk_id: str,
    text: str,
    source_document: str,
    char_start: int = 0,
    char_end: int | None = None,
    heading_path: list[str] | None = None,
    page: int | None = None,
) -> ChunkInput:
    """Build a ``ChunkInput`` from raw fields. Useful for ad-hoc scripts and tests."""
    if char_end is None:
        char_end = char_start + len(text)
    provenance = ChunkProvenance(
        source_document=source_document,
        char_start=char_start,
        char_end=char_end,
        heading_path=list(heading_path or []),
        page=page,
    )
    return ChunkInput(
        chunk_id=chunk_id,
        index=0,
        provenance=provenance,
        text=text,
    )
