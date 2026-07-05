"""Deep-research agent — a LangGraph ReAct loop over the graph-search and
RAG-search tools that already back the regular ``/query`` pipeline.

Unlike ``Dispatcher`` (which routes a single query once, deterministically),
this agent lets the LLM decide, step by step, which tool to call next
(``graph_search`` or ``rag_search``) based on what it has learned so far, up
to a shared tool-call budget (``DEEP_RESEARCH_MAX_TOOL_CALLS``). Once the
model says it has enough evidence — or the budget runs out — a dedicated
synthesis step aggregates every fact collected across all steps into one
cited, structured answer.

Graph shape::

    START -> agent -> (rag_search | graph_search | synthesize)
    rag_search -> agent
    graph_search -> agent
    synthesize -> END

The decision step (``agent``) and the tools never share state beyond the
``_AgentState`` dict LangGraph threads through the graph — no globals, no
hidden coupling to ``Dispatcher`` internals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from graph_reasoning.context_builder import GraphContextBuilder
from graph_reasoning.neo4j_subgraph import Neo4jSubgraphExtractor
from graph_reasoning.reasoner import GraphReasoner
from llm_pipeline_fewshot.llm_parser import YandexGPTError
from routing.query_entity_extractor import QueryEntityExtractor, merge_seed_names

from .rag_client import RAGClient, RAGDocument, StubRAGClient

log = logging.getLogger(__name__)


AGENT_SYSTEM_PROMPT = """Ты — автономный исследовательский агент карты знаний R&D компании
«Норникель» (горно-металлургическая отрасль: гидрометаллургия, пирометаллургия,
флотация, электролиз, обогащение, охрана окружающей среды).

Тебе дан вопрос пользователя. Отвечать на него сразу нельзя — сначала нужно
собрать факты инструментами и только потом передать управление финальному
синтезу. Ты работаешь пошагово: на каждом шаге видишь историю уже сделанных
шагов и должен вернуть ОДНО следующее действие.

Доступные инструменты:

1. graph_search(query) — структурированный обход графа знаний Neo4j.
   Возвращает факты вида «A --uses_material--> B (confidence=0.87,
   source=doc.md, quote="...")», явные противоречия (contradicts) и пробелы
   в знаниях (изолированные сущности, единственный источник). Используй его
   для вопросов о конкретных сущностях (материал/процесс/оборудование/
   эксперимент/публикация), числовых диапазонах и связях между понятиями.
   В запросе указывай конкретные термины предметной области (названия
   материалов, процессов, параметров) — не общие фразы вопроса целиком.

2. rag_search(query) — полнотекстовый поиск (BM25) по корпусу документов и
   графовым рёбрам, отрендеренным в текст. Возвращает фрагменты с указанием
   документа-источника. Используй его для определений, описательного
   контекста, сравнений, причинно-следственных объяснений — всего, что не
   сводится к чистым фактам-триплетам.

Бюджет: у тебя есть ограниченное суммарное число вызовов graph_search +
rag_search на весь запрос (сколько уже потрачено и сколько осталось —
показано в каждом шаге). Трать его экономно:
- не повторяй тот же запрос к тому же инструменту дважды;
- не делай лишний вызов «на всякий случай», если фактов уже достаточно;
- если несколько шагов подряд инструменты не находят ничего релевантного —
  прекращай попытки и переходи к final_answer, а не трать весь бюджет впустую.

Правила:
- Никогда не выдумывай факты, цифры, названия документов или сущностей —
  используй только то, что реально вернул инструмент.
- Предпочитай graph_search для узких фактических/числовых вопросов и
  rag_search для широких/описательных вопросов; для сравнений и вопросов
  «почему» имеет смысл использовать оба инструмента.
- Как только фактов достаточно для полного ответа — немедленно выбирай
  action="final_answer".

Формат ответа — СТРОГО один JSON-объект, без какого-либо текста до или после
него:
{"thought": "1-2 предложения: что уже известно и что решено делать дальше",
 "action": "graph_search" | "rag_search" | "final_answer",
 "action_input": "поисковый запрос для инструмента; пусто, если action=final_answer"}
"""

FINAL_SYSTEM_PROMPT = """Ты — старший аналитик карты знаний R&D «Норникеля». Тебе передан
исходный вопрос пользователя и ВСЕ факты, собранные исследовательским агентом
за несколько шагов поиска по графу знаний и по документам (RAG).

Жёсткие правила:
1. Опирайся ТОЛЬКО на предоставленные факты, графовые рёбра и фрагменты
   документов. Ничего не выдумывай — ни цифр, ни названий, ни источников.
2. Каждый существенный факт подкрепляй ссылкой на источник: имя документа в
   квадратных скобках (например [statya_12.md]) или прямой цитатой из
   графового факта (поле quote в графовом контексте).
3. Если источники противоречат друг другу — явно скажи об этом
   («Противоречие: ...») и укажи оба источника.
4. Если после всех шагов поиска фактов недостаточно для какой-то части
   вопроса — прямо напиши «Недостаточно данных для ...» и перечисли, чего не
   хватает. Не додумывай за источники.
5. Разделяй выводы графа знаний (структурированные факты) и сведения из
   документов (описательный контекст), когда они дают разную гранулярность.
6. Численные значения — с точной величиной и единицей измерения, без
   округления и без выдуманных единиц.

Формат ответа (Markdown):
## Краткий вывод
## Развёрнутый ответ
## Противоречия и пробелы
## Источники
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_ACTIONS = frozenset({"graph_search", "rag_search", "final_answer"})


@dataclass
class ResearchIteration:
    """One tool call made by the agent (either graph_search or rag_search)."""

    index: int
    query: str
    route: str  # "graph_search" | "rag_search"
    confidence: float
    coverage_score: float
    graph_text: str = ""
    rag_documents: list[RAGDocument] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    @property
    def has_graph_evidence(self) -> bool:
        return any(
            marker in self.graph_text
            for marker in ("Facts:", "Knowledge gaps:", "Contradictions:")
        )

    @property
    def evidence_count(self) -> int:
        return (1 if self.has_graph_evidence else 0) + len(self.rag_documents)


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


class _AgentState(TypedDict):
    query: str
    transcript: list[str]
    tool_calls_used: int
    max_tool_calls: int
    iterations: list[ResearchIteration]
    rag_documents: list[RAGDocument]
    graph_snippets: list[str]
    seen_queries: list[str]
    next_action: str
    next_query: str
    final_notes: list[str]
    final_answer: str
    used_llm: bool
    model_uri: str


class DeepResearchAgent:
    """LangGraph ReAct agent over graph_search + rag_search, with capped tool budget.

    Decoupled from ``Dispatcher`` on purpose: it takes the already-built
    ``rag_client`` / ``graph_extractor`` directly (see
    ``Dispatcher.rag_client`` / ``Dispatcher.graph_extractor``) instead of
    reaching into a dispatcher's private attributes.
    """

    def __init__(
        self,
        *,
        rag_client: RAGClient | None = None,
        graph_extractor: Neo4jSubgraphExtractor | None = None,
        llm_client: Any | None = None,
        max_tool_calls: int | None = None,
        max_hops: int = 4,
        max_paths: int = 200,
        rag_max_results: int = 6,
        max_context_chars: int = 24_000,
    ) -> None:
        self._rag_client: RAGClient = rag_client or StubRAGClient()
        self._graph_extractor = graph_extractor
        self._llm_client = llm_client
        self._reasoner = GraphReasoner()
        self._context_builder = GraphContextBuilder()
        self._entity_extractor = QueryEntityExtractor()
        self._max_hops = max_hops
        self._max_paths = max_paths
        self._rag_max_results = max(1, rag_max_results)
        self._max_context_chars = max(1_000, max_context_chars)

        if max_tool_calls is None:
            from config import get_settings
            max_tool_calls = get_settings().deep_research.max_tool_calls
        self._default_max_tool_calls = max(1, int(max_tool_calls))

        self._graph = self._build_graph()

    # ------------------------------------------------------------ public

    async def run(
        self,
        query: str,
        *,
        max_iterations: int | None = None,
    ) -> DeepResearchResult:
        """Run the agent loop. ``max_iterations`` is the total tool-call
        budget for this run (graph_search + rag_search combined); it is
        clamped to the configured ``DEEP_RESEARCH_MAX_TOOL_CALLS`` ceiling
        and never allowed to exceed it. Pass ``None`` to use that ceiling
        directly.
        """
        clean_query = query.strip()
        if not clean_query:
            return DeepResearchResult(
                query=query,
                answer="Пустой запрос: нечего исследовать.",
                used_llm=False,
                notes=("empty_query",),
            )

        budget = (
            self._default_max_tool_calls
            if max_iterations is None
            else max(1, min(int(max_iterations), self._default_max_tool_calls))
        )

        initial_state: _AgentState = {
            "query": clean_query,
            "transcript": [],
            "tool_calls_used": 0,
            "max_tool_calls": budget,
            "iterations": [],
            "rag_documents": [],
            "graph_snippets": [],
            "seen_queries": [],
            "next_action": "",
            "next_query": "",
            "final_notes": [],
            "final_answer": "",
            "used_llm": False,
            "model_uri": "",
        }
        final_state = await self._graph.ainvoke(initial_state)

        iterations: list[ResearchIteration] = final_state["iterations"]
        return DeepResearchResult(
            query=clean_query,
            answer=final_state["final_answer"],
            used_llm=final_state["used_llm"],
            model_uri=final_state["model_uri"],
            iterations=iterations,
            follow_up_queries=[it.query for it in iterations[1:]],
            sources=self._collect_sources(final_state["rag_documents"]),
            notes=tuple(final_state["final_notes"]),
        )

    # -------------------------------------------------------------- graph

    def _build_graph(self):
        graph = StateGraph(_AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("rag_search", self._rag_node)
        graph.add_node("graph_search", self._graph_node)
        graph.add_node("synthesize", self._synthesize_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "rag_search": "rag_search",
                "graph_search": "graph_search",
                "synthesize": "synthesize",
            },
        )
        graph.add_edge("rag_search", "agent")
        graph.add_edge("graph_search", "agent")
        graph.add_edge("synthesize", END)
        return graph.compile()

    @staticmethod
    def _route_after_agent(state: _AgentState) -> str:
        action = state.get("next_action")
        if action in ("rag_search", "graph_search"):
            return action
        return "synthesize"

    # ----------------------------------------------------------- agent node

    async def _agent_node(self, state: _AgentState) -> dict[str, Any]:
        if state["tool_calls_used"] >= state["max_tool_calls"]:
            return {"next_action": "final_answer", "next_query": ""}

        if self._llm_client is None:
            return self._deterministic_next_action(state)

        prompt = self._build_agent_prompt(state)
        try:
            response = await asyncio.to_thread(
                self._llm_client.complete, AGENT_SYSTEM_PROMPT, prompt,
            )
            action, action_query, thought = self._parse_agent_response(response.text)
        except YandexGPTError as exc:
            log.warning(
                "DeepResearchAgent decision call failed: %s: %s", type(exc).__name__, exc,
            )
            return self._deterministic_next_action(state, note=f"llm_error:{type(exc).__name__}")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "DeepResearchAgent unexpected decision error: %s: %s", type(exc).__name__, exc,
            )
            return self._deterministic_next_action(state, note=f"unexpected:{type(exc).__name__}")

        # Guard rail: never let the model answer before it has looked
        # anything up at least once — the system prompt asks for this, but
        # don't rely on the model actually following it.
        if action == "final_answer" and state["tool_calls_used"] == 0:
            action = "graph_search"
            action_query = action_query or state["query"]
            thought = (thought + " (принудительный первый graph_search перед ответом)").strip()

        query_key = f"{action}:{(action_query or state['query']).strip().lower()}"
        update: dict[str, Any] = {"next_action": action, "next_query": action_query}
        if action in ("rag_search", "graph_search") and query_key in state["seen_queries"]:
            # The model asked the same tool the same thing twice — stop
            # instead of burning the rest of the budget on a repeat.
            update["next_action"] = "final_answer"
            update["next_query"] = ""
            thought = f"{thought} (повтор запроса — останавливаюсь)".strip()
        elif action in ("rag_search", "graph_search"):
            update["seen_queries"] = state["seen_queries"] + [query_key]

        step_no = state["tool_calls_used"] + 1
        update["transcript"] = state["transcript"] + [
            f"Шаг {step_no}: мысль: {thought or '—'} | действие: {update['next_action']}"
            f"({update['next_query']!r})"
        ]
        return update

    def _deterministic_next_action(
        self, state: _AgentState, note: str | None = None,
    ) -> dict[str, Any]:
        """No-LLM fallback policy: one graph_search, then one rag_search, then stop.

        Used when no LLM client is configured (mock mode) so the agent still
        does something useful offline instead of failing outright.
        """
        used = state["tool_calls_used"]
        action = "graph_search" if used == 0 else "rag_search" if used == 1 else "final_answer"
        update: dict[str, Any] = {"next_action": action, "next_query": state["query"]}
        if note:
            update["final_notes"] = state["final_notes"] + [note]
        return update

    def _build_agent_prompt(self, state: _AgentState) -> str:
        lines = [f"# Исходный вопрос пользователя\n{state['query']}\n"]
        lines.append(
            f"# Бюджет инструментов\nИспользовано: {state['tool_calls_used']} из "
            f"{state['max_tool_calls']}.\n"
        )
        if state["transcript"]:
            lines.append("# История твоих предыдущих шагов")
            lines.extend(state["transcript"])
            lines.append("")
        else:
            lines.append("# История\nПока не было ни одного шага.\n")
        lines.append(
            "Реши следующий шаг. Ответь ОДНИМ JSON-объектом без пояснений вокруг него:\n"
            '{"thought": "...", "action": "graph_search" | "rag_search" | "final_answer", '
            '"action_input": "..."}'
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_agent_response(text: str) -> tuple[str, str, str]:
        match = _JSON_BLOCK_RE.search(text or "")
        if not match:
            return "final_answer", "", (text or "").strip()[:300]
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return "final_answer", "", (text or "").strip()[:300]
        action = str(payload.get("action") or "final_answer").strip().lower()
        if action not in _VALID_ACTIONS:
            action = "final_answer"
        action_query = str(payload.get("action_input") or payload.get("query") or "").strip()
        thought = str(payload.get("thought") or "").strip()
        return action, action_query, thought

    # ------------------------------------------------------------ tool nodes

    async def _rag_node(self, state: _AgentState) -> dict[str, Any]:
        q = (state["next_query"] or state["query"]).strip()
        seeds = merge_seed_names(self._entity_extractor.extract(q)) or None
        try:
            result = await self._rag_client.retrieve(
                q, entity_filter=seeds, max_results=self._rag_max_results,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("DeepResearchAgent rag_search failed: %s: %s", type(exc).__name__, exc)
            result_documents: list[RAGDocument] = []
            notes: tuple[str, ...] = (f"rag_error:{type(exc).__name__}",)
        else:
            result_documents = list(result.documents)
            notes = tuple(result.notes)

        existing_ids = {doc.doc_id for doc in state["rag_documents"] if doc.doc_id}
        new_docs = [d for d in result_documents if not d.doc_id or d.doc_id not in existing_ids]

        step_no = state["tool_calls_used"] + 1
        iteration = ResearchIteration(
            index=step_no,
            query=q,
            route="rag_search",
            confidence=1.0 if result_documents else 0.0,
            coverage_score=round(step_no / state["max_tool_calls"], 4),
            rag_documents=result_documents,
            notes=notes,
        )
        observation = self._render_rag_observation(result_documents)
        return {
            "tool_calls_used": step_no,
            "rag_documents": state["rag_documents"] + new_docs,
            "iterations": state["iterations"] + [iteration],
            "transcript": state["transcript"] + [f"Наблюдение (RAG, запрос {q!r}): {observation}"],
        }

    async def _graph_node(self, state: _AgentState) -> dict[str, Any]:
        q = (state["next_query"] or state["query"]).strip()
        graph_text, found = await self._run_graph_search(q)

        step_no = state["tool_calls_used"] + 1
        iteration = ResearchIteration(
            index=step_no,
            query=q,
            route="graph_search",
            confidence=1.0 if found else 0.0,
            coverage_score=round(step_no / state["max_tool_calls"], 4),
            graph_text=graph_text,
        )
        snippets = state["graph_snippets"] + [graph_text] if found else state["graph_snippets"]
        return {
            "tool_calls_used": step_no,
            "graph_snippets": snippets,
            "iterations": state["iterations"] + [iteration],
            "transcript": state["transcript"] + [
                f"Наблюдение (граф, запрос {q!r}): {self._truncate_observation(graph_text)}"
            ],
        }

    async def _run_graph_search(self, query: str) -> tuple[str, bool]:
        if self._graph_extractor is None:
            return "Граф недоступен: Neo4j не подключён.", False
        seeds = merge_seed_names(self._entity_extractor.extract(query))
        if not seeds:
            return f"Не удалось выделить сущности-сиды из запроса «{query}».", False
        try:
            context = await self._graph_extractor.extract_subgraph(
                seed_entity_names=seeds, max_hops=self._max_hops, limit=self._max_paths,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("DeepResearchAgent graph_search failed: %s: %s", type(exc).__name__, exc)
            return f"Ошибка запроса к графу: {type(exc).__name__}", False
        self._reasoner.enrich_context(context)
        text = self._context_builder.build_text_context(context)
        return text, bool(context.edges)

    @staticmethod
    def _render_rag_observation(documents: list[RAGDocument]) -> str:
        if not documents:
            return "ничего не найдено."
        parts = [
            f"{doc.title} (score={doc.score:.2f}, source={doc.source or doc.doc_id})"
            for doc in documents[:5]
        ]
        return "; ".join(parts)

    @staticmethod
    def _truncate_observation(text: str, limit: int = 400) -> str:
        text = (text or "").strip()
        if not text:
            return "пусто."
        return text[:limit] + ("…" if len(text) > limit else "")

    # -------------------------------------------------------- synthesis node

    async def _synthesize_node(self, state: _AgentState) -> dict[str, Any]:
        if self._llm_client is None:
            return {
                "final_answer": self._fallback_answer(state),
                "used_llm": False,
                "model_uri": "",
                "final_notes": state["final_notes"] + ["no_llm_client"],
            }

        prompt = self._build_synthesis_prompt(state)
        try:
            response = await asyncio.to_thread(
                self._llm_client.complete, FINAL_SYSTEM_PROMPT, prompt,
            )
            return {
                "final_answer": response.text.strip(),
                "used_llm": True,
                "model_uri": str(self._llm_client.model_uri),
            }
        except YandexGPTError as exc:
            log.warning("DeepResearchAgent synthesis failed: %s: %s", type(exc).__name__, exc)
            return {
                "final_answer": self._fallback_answer(state),
                "used_llm": False,
                "model_uri": str(getattr(self._llm_client, "model_uri", "")),
                "final_notes": state["final_notes"] + [f"llm_error:{type(exc).__name__}"],
            }
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "DeepResearchAgent unexpected synthesis error: %s: %s", type(exc).__name__, exc,
            )
            return {
                "final_answer": self._fallback_answer(state),
                "used_llm": False,
                "model_uri": str(getattr(self._llm_client, "model_uri", "")),
                "final_notes": state["final_notes"] + [f"unexpected:{type(exc).__name__}"],
            }

    def _build_synthesis_prompt(self, state: _AgentState) -> str:
        sections = [f"# Исходный вопрос\n{state['query']}\n"]
        sections.append(f"# Ход расследования ({state['tool_calls_used']} вызовов инструментов)")
        sections.extend(state["transcript"])
        sections.append("")

        if state["graph_snippets"]:
            sections.append("# Факты из графа знаний (все шаги)")
            sections.append("```")
            sections.append(
                self._truncate("\n\n".join(state["graph_snippets"]), limit=self._max_context_chars // 2)
            )
            sections.append("```")
            sections.append("")

        if state["rag_documents"]:
            sections.append("# Документы (RAG, все шаги)")
            for idx, doc in enumerate(state["rag_documents"], start=1):
                sections.append(
                    f"{idx}. [{doc.source or doc.doc_id}] **{doc.title}** (score={doc.score:.2f})\n"
                    f"   {doc.snippet[:600]}"
                )
            sections.append("")

        sections.append("# Задача")
        sections.append(
            "Дай развёрнутый, структурированный ответ на исходный вопрос, используя только "
            "факты выше. Процитируй документы/графовые источники, на которые опирался ответ, "
            "и явно укажи пробелы, если что-то осталось невыясненным."
        )
        return self._truncate("\n".join(sections))

    def _fallback_answer(self, state: _AgentState) -> str:
        lines: list[str] = [
            "## Краткий вывод",
            "LLM-синтез недоступен, показан агрегированный результат агентского поиска.",
            "",
            "## Итерации",
        ]
        for iteration in state["iterations"]:
            lines.append(
                f"- Шаг {iteration.index}: `{iteration.route}` (запрос {iteration.query!r}), "
                f"RAG документов={len(iteration.rag_documents)}, "
                f"граф={'есть' if iteration.has_graph_evidence else 'нет'}."
            )

        docs = self._collect_documents(state["rag_documents"])
        lines.append("")
        lines.append("## Документы")
        if not docs:
            lines.append("Документы не найдены.")
        for doc in docs[:12]:
            lines.append(f"- **{doc.title}** [{doc.source or doc.doc_id}]: {doc.snippet[:240]}")

        if state["graph_snippets"]:
            lines.append("")
            lines.append("## Графовые факты")
            lines.append("```")
            lines.append(self._truncate("\n\n".join(state["graph_snippets"]), limit=8_000))
            lines.append("```")

        return "\n".join(lines).rstrip() + "\n"

    def _truncate(self, text: str, limit: int | None = None) -> str:
        max_chars = limit or self._max_context_chars
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n" + text[-half:]

    @staticmethod
    def _collect_documents(rag_documents: list[RAGDocument]) -> list[RAGDocument]:
        return sorted(rag_documents, key=lambda doc: doc.score, reverse=True)

    @staticmethod
    def _collect_sources(rag_documents: list[RAGDocument]) -> list[str]:
        sources: list[str] = []
        for doc in DeepResearchAgent._collect_documents(rag_documents):
            label = doc.source or doc.title or doc.doc_id
            if label and label not in sources:
                sources.append(label)
        return sources
