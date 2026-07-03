"""
LLM answer synthesizer — the final layer of the pipeline.

Takes a ``DispatchResult`` (routing decision + graph context +
optional RAG documents) and produces a natural-language answer
to the user's query. Calls Yandex Foundation Models under the
hood.

If the LLM client is unavailable (no API key, network error,
etc.), the synthesizer falls back to a deterministic
"best-effort" rendering of the graph context — useful for
offline demos and tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from llm_pipeline_fewshot.llm_parser import YandexGPTClient, YandexGPTError

if TYPE_CHECKING:  # avoid circular import at runtime
    from .dispatcher import DispatchResult

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """Ты — старший аналитик графа знаний R&D Норникеля (горно-металлургическая отрасль).
Твоя задача — дать структурированный, проверяемый ответ на русском языке на основе
предоставленных фактов из графа знаний и/или фрагментов документов.

Жёсткие правила:
1. Опирайся ТОЛЬКО на предоставленные факты. Не додумывай.
2. Каждый существенный факт подкрепи цитатой (поля `quote`) или указанием источника.
3. Если факты противоречат друг другу — отметь это явно ("Противоречие: ...").
4. Если данных недостаточно — скажи "В графе знаний и документах недостаточно данных"
   и перечисли, какие именно пробелы (`knowledge gaps`).
5. Если вопрос определительный ("что такое X?") — дай развёрнутое определение со ссылками.
6. Если вопрос про сравнение — дай таблицу или явное A vs Б.
7. Если есть числовые ограничения в запросе — учти их.
8. Будь кратким, но не жертвуй точностью. Длина ответа — соразмерна объёму фактов.
9. В конце перечисли "Источники" — уникальные документы, на которые ты опирался.

Формат ответа (Markdown):
- Краткий вывод (1–3 предложения)
- Подробности (список или таблица)
- Противоречия / пробелы (если есть)
- Источники
"""


@dataclass
class SynthesisResult:
    answer: str
    used_llm: bool = False
    model_uri: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    notes: tuple[str, ...] = ()


class AnswerSynthesizer:
    """Synthesizes a natural-language answer from a ``DispatchResult``."""

    def __init__(
        self,
        client: YandexGPTClient | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_context_chars: int = 12_000,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._max_context_chars = max(100, max_context_chars)

    async def synthesize(self, dispatch_result: "DispatchResult") -> SynthesisResult:
        user_prompt = self._build_user_prompt(dispatch_result)

        if self._client is None:
            return self._fallback(dispatch_result, reason="no_llm_client")

        try:
            response = self._client.complete(self._system_prompt, user_prompt)
            return SynthesisResult(
                answer=response.text.strip(),
                used_llm=True,
                model_uri=self._client.model_uri,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except YandexGPTError as exc:
            log.warning(
                "AnswerSynthesizer LLM call failed: %s: %s — falling back to context render",
                type(exc).__name__,
                exc,
            )
            return self._fallback(
                dispatch_result,
                reason=f"llm_error:{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("AnswerSynthesizer unexpected error: %s: %s", type(exc).__name__, exc)
            return self._fallback(
                dispatch_result,
                reason=f"unexpected:{type(exc).__name__}: {exc}",
            )

    # ---------------------------------------------------------------- helpers

    def _build_user_prompt(self, dispatch_result: DispatchResult) -> str:
        decision = dispatch_result.decision
        sections: list[str] = []
        sections.append(f"# Запрос пользователя\n{dispatch_result.query}\n")

        sections.append("# Решение роутера")
        sections.append(f"- Маршрут: `{decision.route.value}`")
        sections.append(f"- Уверенность: {decision.confidence:.2f}")
        sections.append(f"- Coverage: {decision.coverage_score:.2f}, Ambiguity: {decision.ambiguity_score:.2f}")
        if decision.reasons:
            sections.append(f"- Причины: {', '.join(decision.reasons[:5])}")
        sections.append("")

        if dispatch_result.graph_text:
            text = self._truncate(dispatch_result.graph_text)
            sections.append("# Контекст из графа знаний")
            sections.append("```")
            sections.append(text)
            sections.append("```")
            sections.append("")

        if dispatch_result.decision.graph_coverage is not None:
            cov = dispatch_result.decision.graph_coverage
            if cov.has_contradictions or cov.has_knowledge_gaps:
                sections.append("# Замечания по покрытию графа")
                if cov.contradiction_count:
                    sections.append(f"- Противоречий: {cov.contradiction_count}")
                if cov.gap_count:
                    sections.append(f"- Пробелов в знаниях: {cov.gap_count}")
                if cov.unmatched_seed_names:
                    sections.append(
                        f"- Сиды не найдены в графе: {', '.join(cov.unmatched_seed_names[:5])}"
                    )
                sections.append("")

        if dispatch_result.rag_result is not None and dispatch_result.rag_result.documents:
            sections.append("# Документы (RAG)")
            for index, doc in enumerate(dispatch_result.rag_result.documents[:5], start=1):
                sections.append(
                    f"{index}. **{doc.title}** (score={doc.score:.2f})\n"
                    f"   Источник: {doc.source}\n"
                    f"   {doc.snippet[:500]}"
                )
            sections.append("")

        sections.append("# Инструкция")
        sections.append(
            "Дай структурированный ответ на запрос пользователя, используя факты выше. "
            "Если фактов недостаточно — явно укажи это. "
            "Ссылайся на источники в формате [doc.md] или [eX]."
        )

        user_prompt = "\n".join(sections)
        return self._truncate(user_prompt)

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_context_chars:
            return text
        half = self._max_context_chars // 2
        return (
            text[:half]
            + f"\n\n... [{len(text) - self._max_context_chars} chars truncated] ...\n\n"
            + text[-half:]
        )

    def _fallback(self, dispatch_result: DispatchResult, *, reason: str) -> SynthesisResult:
        """Render the dispatch result as a structured Markdown 'best-effort' answer."""
        decision = dispatch_result.decision
        lines: list[str] = []

        if decision.route.value == "no_data":
            lines.append("## Краткий вывод")
            lines.append(
                "В графе знаний и подключённых документах недостаточно данных для ответа на этот запрос."
            )
            if decision.reasons:
                lines.append(f"_Причины: {', '.join(decision.reasons)}_")
            return SynthesisResult(
                answer="\n".join(lines),
                used_llm=False,
                error=reason,
                notes=("fallback_no_data",),
            )

        lines.append("## Краткий вывод")
        lines.append(
            f"Решение роутера: `{decision.route.value}` "
            f"(уверенность {decision.confidence:.2f}, "
            f"coverage {decision.coverage_score:.2f})."
        )
        lines.append("")

        if dispatch_result.graph_text:
            lines.append("## Найденные факты (граф знаний)")
            lines.append("")
            lines.append("```")
            lines.append(self._truncate(dispatch_result.graph_text))
            lines.append("```")
            lines.append("")

        if dispatch_result.rag_result is not None and dispatch_result.rag_result.documents:
            lines.append("## Документы (RAG)")
            for index, doc in enumerate(dispatch_result.rag_result.documents[:5], start=1):
                lines.append(
                    f"{index}. **{doc.title}** (score={doc.score:.2f}, source={doc.source})"
                )
                lines.append(f"   {doc.snippet[:300]}")
            lines.append("")

        if dispatch_result.decision.graph_coverage is not None:
            cov = dispatch_result.decision.graph_coverage
            notes: list[str] = []
            if cov.contradiction_count:
                notes.append(f"⚠ {cov.contradiction_count} противоречий в графе")
            if cov.gap_count:
                notes.append(f"⚠ {cov.gap_count} пробелов в знаниях")
            if notes:
                lines.append("## Замечания")
                for note in notes:
                    lines.append(f"- {note}")
                lines.append("")

        lines.append("---")
        lines.append(
            f"_LLM-синтез недоступен ({reason}); показан структурированный дамп контекста._"
        )

        return SynthesisResult(
            answer="\n".join(lines).rstrip() + "\n",
            used_llm=False,
            error=reason,
            notes=("fallback_render",),
        )


def attach_synthesis(
    dispatch_result: "DispatchResult",
    synthesis: SynthesisResult,
) -> "DispatchResult":
    """Return a new ``DispatchResult`` with the synthesized answer attached as a note.

    Keeps the original dispatch result immutable (the underlying dataclass
    isn't frozen, but treating it as such makes the call sites clearer).
    """
    if synthesis.notes:
        notes = dispatch_result.notes + ("synthesis:" + ";".join(synthesis.notes),)
    else:
        notes = dispatch_result.notes
    return DispatchResult(
        query=dispatch_result.query,
        decision=dispatch_result.decision,
        graph_context=dispatch_result.graph_context,
        graph_text=dispatch_result.graph_text,
        rag_result=dispatch_result.rag_result,
        notes=notes,
    )
