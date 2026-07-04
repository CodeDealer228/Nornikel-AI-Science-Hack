"""Iterative deep-research agent over graph and RAG tools."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from llm_pipeline_fewshot.llm_parser import YandexGPTError

from .dispatcher import DispatchResult, Dispatcher
from .rag_client import RAGDocument

log = logging.getLogger(__name__)


DEEP_RESEARCH_SYSTEM_PROMPT = """Ты — исследовательский агент R&D карты знаний Норникеля.
Тебе дан исходный вопрос и результаты нескольких итераций поиска по графу знаний и RAG.

Правила:
1. Отвечай только по предоставленным фактам, графовым рёбрам и фрагментам документов.
2. Не выдумывай факты, численные значения, авторов, технологии и источники.
3. Если фактов недостаточно, явно укажи пробелы.
4. Разделяй выводы из графа знаний и сведения из документов, если они различаются.
5. В конце перечисли источники документов, на которые опирался ответ.

Формат Markdown:
- Краткий вывод
- Найденные факты
- Детали по документам и графу
- Противоречия и пробелы
- Источники
"""


@dataclass
class ResearchIteration:
    index: int
    query: str
    route: str
    confidence: float
    coverage_score: float
    graph_text: str = ""
    rag_documents: list[RAGDocument] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    @property
    def evidence_count(self) -> int:
        graph_hit = 1 if self.has_graph_evidence else 0
        return graph_hit + len(self.rag_documents)

    @property
    def has_graph_evidence(self) -> bool:
        return any(
            marker in self.graph_text
            for marker in ("Facts:", "Knowledge gaps:", "Contradictions:")
        )


@dataclass
class DeepResearchResult:
    query: str
    answer: str
    used_llm: bool
    model_uri: str = ""
    iterations: list[ResearchIteration] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    notes: tuple[str, ...] = ()


class DeepResearchAgent:
    """Runs several graph+RAG tool calls before final answer synthesis."""

    def __init__(
        self,
        dispatcher: Dispatcher,
        *,
        llm_client: Any | None = None,
        max_context_chars: int = 24_000,
    ) -> None:
        self._dispatcher = dispatcher
        self._llm_client = llm_client
        self._max_context_chars = max(1_000, max_context_chars)

    async def run(
        self,
        query: str,
        *,
        max_iterations: int = 3,
    ) -> DeepResearchResult:
        clean_query = query.strip()
        if not clean_query:
            return DeepResearchResult(
                query=query,
                answer="Пустой запрос: нечего исследовать.",
                used_llm=False,
                notes=("empty_query",),
            )

        iteration_limit = max(1, min(max_iterations, 5))
        planned_queries = [clean_query]
        iterations: list[ResearchIteration] = []
        seen_queries: set[str] = set()
        seen_doc_ids: set[str] = set()
        stagnant_steps = 0

        for index in range(1, iteration_limit + 1):
            current_query = self._next_query(planned_queries, seen_queries)
            if current_query is None:
                break
            seen_queries.add(self._query_key(current_query))

            iteration = await self._run_iteration(index, current_query)
            iterations.append(iteration)

            new_docs = {
                doc.doc_id
                for doc in iteration.rag_documents
                if doc.doc_id and doc.doc_id not in seen_doc_ids
            }
            seen_doc_ids.update(new_docs)
            if not new_docs and iteration.evidence_count == 0:
                stagnant_steps += 1
            else:
                stagnant_steps = 0

            if index < iteration_limit:
                planned_queries.extend(
                    self._derive_follow_up_queries(clean_query, iteration)
                )
            if stagnant_steps >= 2:
                break

        answer, used_llm, model_uri, notes = await self._synthesize(clean_query, iterations)
        sources = self._collect_sources(iterations)
        return DeepResearchResult(
            query=clean_query,
            answer=answer,
            used_llm=used_llm,
            model_uri=model_uri,
            iterations=iterations,
            follow_up_queries=[it.query for it in iterations[1:]],
            sources=sources,
            notes=notes,
        )

    async def _run_iteration(self, index: int, query: str) -> ResearchIteration:
        decision = await self._dispatcher._router.route(query)  # noqa: SLF001
        graph_context, rag_result = await asyncio.gather(
            self._dispatcher._graph_query(decision),  # noqa: SLF001
            self._dispatcher._rag_query(query, decision),  # noqa: SLF001
        )
        graph_text = self._dispatcher._context_builder.build_text_context(  # noqa: SLF001
            graph_context,
        )
        return ResearchIteration(
            index=index,
            query=query,
            route=str(decision.route.value),
            confidence=round(decision.confidence, 4),
            coverage_score=round(decision.coverage_score, 4),
            graph_text=graph_text,
            rag_documents=list(rag_result.documents),
            notes=tuple(rag_result.notes),
        )

    async def _synthesize(
        self,
        query: str,
        iterations: list[ResearchIteration],
    ) -> tuple[str, bool, str, tuple[str, ...]]:
        prompt = self._build_prompt(query, iterations)
        if self._llm_client is None:
            return self._fallback_answer(query, iterations), False, "", ("no_llm_client",)

        try:
            response = await asyncio.to_thread(
                self._llm_client.complete,
                DEEP_RESEARCH_SYSTEM_PROMPT,
                prompt,
            )
            return response.text.strip(), True, str(self._llm_client.model_uri), ()
        except YandexGPTError as exc:
            log.warning(
                "DeepResearchAgent LLM call failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return (
                self._fallback_answer(query, iterations),
                False,
                str(getattr(self._llm_client, "model_uri", "")),
                (f"llm_error:{type(exc).__name__}", str(exc)[:300]),
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("DeepResearchAgent unexpected LLM error: %s: %s", type(exc).__name__, exc)
            return (
                self._fallback_answer(query, iterations),
                False,
                str(getattr(self._llm_client, "model_uri", "")),
                (f"unexpected:{type(exc).__name__}", str(exc)[:300]),
            )

    def _build_prompt(self, query: str, iterations: list[ResearchIteration]) -> str:
        sections: list[str] = [f"# Исходный вопрос\n{query}\n"]
        sections.append("# Итерации исследования")
        for iteration in iterations:
            sections.append(
                f"## Итерация {iteration.index}: {iteration.query}\n"
                f"- route: {iteration.route}\n"
                f"- confidence: {iteration.confidence:.2f}\n"
                f"- coverage: {iteration.coverage_score:.2f}\n"
            )
            if iteration.graph_text.strip():
                sections.append("### Граф знаний")
                sections.append("```")
                sections.append(iteration.graph_text[:6_000])
                sections.append("```")
            if iteration.rag_documents:
                sections.append("### RAG документы")
                for idx, doc in enumerate(iteration.rag_documents[:6], start=1):
                    sections.append(
                        f"{idx}. [{doc.doc_id}] **{doc.title}** score={doc.score:.2f}\n"
                        f"   source={doc.source}\n"
                        f"   snippet={doc.snippet[:900]}"
                    )
            if iteration.notes:
                sections.append(f"### Notes\n{', '.join(iteration.notes)}")
            sections.append("")

        sections.append("# Задача")
        sections.append(
            "Собери финальный ответ на исходный вопрос. Учитывай все итерации, "
            "не повторяй одинаковые факты и явно укажи источники."
        )
        return self._truncate("\n".join(sections))

    def _fallback_answer(self, query: str, iterations: list[ResearchIteration]) -> str:
        lines: list[str] = [
            "## Краткий вывод",
            "LLM-синтез недоступен, показан агрегированный результат агентского поиска.",
            "",
            "## Итерации",
        ]
        for iteration in iterations:
            lines.append(
                f"- Итерация {iteration.index}: `{iteration.route}`, "
                f"confidence={iteration.confidence:.2f}, coverage={iteration.coverage_score:.2f}, "
                f"RAG документов={len(iteration.rag_documents)}, "
                f"граф={'есть' if iteration.has_graph_evidence else 'нет'}."
            )

        lines.append("")
        lines.append("## Документы")
        docs = self._unique_documents(iterations)
        if not docs:
            lines.append("Документы не найдены.")
        for doc in docs[:12]:
            lines.append(f"- **{doc.title}** score={doc.score:.2f}: {doc.snippet[:240]}")

        graph_blocks = [it.graph_text for it in iterations if it.graph_text.strip()]
        if graph_blocks:
            lines.append("")
            lines.append("## Графовые факты")
            lines.append("```")
            lines.append(self._truncate("\n\n".join(graph_blocks), limit=8_000))
            lines.append("```")

        return "\n".join(lines).rstrip() + "\n"

    def _derive_follow_up_queries(
        self,
        original_query: str,
        iteration: ResearchIteration,
    ) -> list[str]:
        candidates: list[str] = []
        for doc in iteration.rag_documents[:4]:
            title_terms = self._extract_terms(doc.title)
            if title_terms:
                candidates.append(f"{original_query} {' '.join(title_terms[:4])}")

        graph_terms = self._extract_terms(iteration.graph_text)
        if graph_terms:
            candidates.append(f"{original_query} {' '.join(graph_terms[:5])}")

        if iteration.route == "graph_only":
            candidates.append(f"{original_query} документы публикация эксперимент")
        elif iteration.route == "rag_only":
            candidates.append(f"{original_query} сущности связи параметры результаты")

        return candidates

    @staticmethod
    def _extract_terms(text: str) -> list[str]:
        stop = {
            "какие", "какая", "какой", "что", "где", "когда", "между", "через",
            "для", "при", "или", "если", "это", "source", "quote", "confidence",
            "graph", "context", "facts",
        }
        terms: list[str] = []
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9+\-/]{4,}", text):
            lowered = token.lower()
            if lowered in stop or lowered.isdigit():
                continue
            if lowered not in terms:
                terms.append(lowered)
            if len(terms) >= 12:
                break
        return terms

    @staticmethod
    def _next_query(planned_queries: list[str], seen_queries: set[str]) -> str | None:
        while planned_queries:
            candidate = planned_queries.pop(0).strip()
            if candidate and DeepResearchAgent._query_key(candidate) not in seen_queries:
                return candidate
        return None

    @staticmethod
    def _query_key(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    @staticmethod
    def _collect_sources(iterations: list[ResearchIteration]) -> list[str]:
        sources: list[str] = []
        for doc in DeepResearchAgent._unique_documents(iterations):
            label = doc.source or doc.title or doc.doc_id
            if label and label not in sources:
                sources.append(label)
        return sources

    @staticmethod
    def _unique_documents(iterations: list[ResearchIteration]) -> list[RAGDocument]:
        docs: list[RAGDocument] = []
        seen: set[str] = set()
        for iteration in iterations:
            for doc in iteration.rag_documents:
                key = doc.doc_id or doc.title or doc.snippet[:80]
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
        return sorted(docs, key=lambda item: item.score, reverse=True)

    def _truncate(self, text: str, limit: int | None = None) -> str:
        max_chars = limit or self._max_context_chars
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n" + text[-half:]
