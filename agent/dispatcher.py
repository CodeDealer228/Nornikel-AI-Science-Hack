"""
Query dispatcher — the execution layer that turns a
``QueryRoutingDecision`` into an actionable response.

Inputs:  user query (str)
Outputs: ``DispatchResult`` carrying the routing decision,
         graph context (when applicable), RAG result (when
         applicable), and a human-readable Markdown summary.

The dispatcher does NOT generate the final natural-language
answer. That is the job of the LLM-synthesis layer (planned for
a later step). The dispatcher simply prepares the context that
the synthesis layer will consume.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from graph_reasoning.context_builder import GraphContextBuilder
from graph_reasoning.models import GraphReasoningContext
from graph_reasoning.neo4j_subgraph import Neo4jSubgraphExtractor
from graph_reasoning.reasoner import GraphReasoner
from routing.query_models import (
    GraphCoverageReport,
    QueryAnalysis,
    QueryRoute,
    QueryRoutingDecision,
)

from .rag_client import RAGClient, RAGResult, StubRAGClient
from .synthesizer import AnswerSynthesizer, SynthesisResult

log = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """All artefacts produced by the dispatcher for one user query."""

    query: str
    decision: QueryRoutingDecision
    graph_context: GraphReasoningContext | None = None
    graph_text: str | None = None
    rag_result: RAGResult | None = None
    synthesis: SynthesisResult | None = None
    notes: tuple[str, ...] = ()

    def to_markdown(self) -> str:
        """Render the dispatch result as a human-readable Markdown summary.

        The synthesized answer (if present) is shown at the top so that
        ``agent.cli`` and other consumers can read the final user-facing
        text first, followed by the underlying routing decision and
        context for transparency.
        """
        lines: list[str] = []
        decision = self.decision
        if self.synthesis is not None:
            lines.append(self.synthesis.answer.rstrip())
            lines.append("")
            if self.synthesis.used_llm:
                lines.append(
                    f"_Синтезировано через {self.synthesis.model_uri} "
                    f"(input={self.synthesis.input_tokens}, "
                    f"output={self.synthesis.output_tokens})._"
                )
            else:
                lines.append(
                    f"_LLM-синтез не использовался: {self.synthesis.error or 'fallback'}_"
                )
            lines.append("")
            lines.append("---")
            lines.append("")
        lines.append(f"# Dispatch detail for: {self.query!r}")
        lines.append("")
        lines.append(f"- **Route:** `{decision.route.value}`")
        lines.append(f"- **Confidence:** {decision.confidence:.2f}")
        lines.append(f"- **Coverage score:** {decision.coverage_score:.2f}")
        lines.append(f"- **Ambiguity score:** {decision.ambiguity_score:.2f}")
        if decision.reasons:
            lines.append(f"- **Reasons:** {', '.join(decision.reasons)}")
        lines.append("")
        if decision.query_analysis is not None:
            analysis = decision.query_analysis
            markers: list[str] = []
            if analysis.has_numeric_constraint:
                markers.append("numeric")
            if analysis.has_geo_marker:
                markers.append("geography")
            if analysis.has_temporal_marker:
                markers.append("temporal")
            if markers:
                lines.append(f"- **Query markers:** {', '.join(markers)}")
            if analysis.seed_entities:
                seeds = ", ".join(
                    f"{ent.surface}→{ent.canonical}" for ent in analysis.seed_entities
                )
                lines.append(f"- **Seed entities:** {seeds}")
            lines.append("")
        if decision.graph_coverage is not None:
            cov = decision.graph_coverage
            lines.append("## Graph coverage")
            lines.append(
                f"- matched seeds: {len(cov.matched_seed_names)}/{len(cov.seed_entities)}"
            )
            lines.append(f"- nodes: {cov.total_nodes}, edges: {cov.total_edges}")
            lines.append(f"- max hop observed: {cov.max_hop_observed}")
            if cov.has_contradictions:
                lines.append(f"- contradictions: {cov.contradiction_count}")
            if cov.has_knowledge_gaps:
                lines.append(f"- knowledge gaps: {cov.gap_count}")
            lines.append("")
        if self.graph_text:
            lines.append("## Graph context")
            lines.append("")
            lines.append("```")
            lines.append(self.graph_text)
            lines.append("```")
            lines.append("")
        if self.rag_result is not None:
            rag = self.rag_result
            lines.append("## RAG retrieval")
            if rag.notes:
                lines.append(f"_Notes: {', '.join(rag.notes)}_")
            if not rag.documents:
                lines.append("_No RAG documents retrieved._")
            else:
                for index, doc in enumerate(rag.documents, start=1):
                    lines.append(f"{index}. **{doc.title}** (score={doc.score:.2f})")
                    if doc.source:
                        lines.append(f"   - source: {doc.source}")
                    if doc.matched_entities:
                        lines.append(
                            f"   - matched entities: {', '.join(doc.matched_entities)}"
                        )
                    lines.append(f"   - {doc.snippet}")
            lines.append("")
        if self.notes:
            lines.append(f"## Notes")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


class Dispatcher:
    """Executes a routing decision and prepares the context the LLM-synthesis
    layer will consume.

    The dispatcher is intentionally backend-agnostic:
    * The graph backend is provided as a ``Neo4jSubgraphExtractor``
      (the same one used by the ``GraphCoverageAnalyzer``).
    * The RAG backend is any object implementing the ``RAGClient`` protocol.
      A ``StubRAGClient`` is used by default so the dispatcher is usable
      before the real RAG is integrated.
    """

    def __init__(
        self,
        router: Any,
        graph_extractor: Neo4jSubgraphExtractor | None = None,
        reasoner: GraphReasoner | None = None,
        rag_client: RAGClient | None = None,
        synthesizer: AnswerSynthesizer | None = None,
        max_hops: int = 4,
        max_paths: int = 200,
        rag_max_results: int = 8,
    ) -> None:
        self._router = router
        self._graph_extractor = graph_extractor
        self._reasoner = reasoner or GraphReasoner()
        self._context_builder = GraphContextBuilder()
        self._rag_client: RAGClient = rag_client or StubRAGClient()
        self._synthesizer = synthesizer
        # Clamp hops to [1, 4]. The upper bound matches the contract in
        # ``GraphCoverageAnalyzer``; if the caller passes more, the value is
        # silently ignored — surface that so tuning doesn't get lost in
        # dev/staging.
        if max_hops < 1 or max_hops > 4:
            log.warning(
                "Dispatcher max_hops=%d is outside the supported [1, 4] range; clamping.",
                max_hops,
            )
        self._max_hops = max(1, min(max_hops, 4))
        self._max_paths = max(1, max_paths)
        self._rag_max_results = max(1, rag_max_results)

    # ------------------------------------------------------------- accessors

    @property
    def rag_client(self) -> RAGClient:
        """The configured RAG backend (stub or real) — reused by other agents."""
        return self._rag_client

    @property
    def graph_extractor(self) -> Neo4jSubgraphExtractor | None:
        """The Neo4j subgraph extractor, or ``None`` when offline."""
        return self._graph_extractor

    @property
    def max_hops(self) -> int:
        return self._max_hops

    @property
    def max_paths(self) -> int:
        return self._max_paths

    async def dispatch(
        self,
        query: str,
        *,
        synthesize: bool = True,
    ) -> DispatchResult:
        """Route a query and execute the corresponding path.

        When ``synthesize`` is True (the default) and a synthesizer is
        configured, the dispatcher runs the LLM synthesis after the
        decision path completes and attaches the resulting answer to
        the ``DispatchResult`` as ``synthesis``.
        """
        if not query or not query.strip():
            decision = await self._router.route(query)
            result = DispatchResult(
                query=query,
                decision=decision,
                notes=("empty_query",),
            )
            return await self._maybe_synthesize(result, synthesize)

        decision = await self._router.route(query)
        notes: list[str] = []

        if decision.route == QueryRoute.NO_DATA:
            result = DispatchResult(
                query=query,
                decision=decision,
                notes=("no_data_path",),
            )
            return await self._maybe_synthesize(result, synthesize)

        if decision.route == QueryRoute.GRAPH_ONLY:
            graph_context = await self._graph_query(decision)
            graph_text = self._context_builder.build_text_context(graph_context)
            result = DispatchResult(
                query=query,
                decision=decision,
                graph_context=graph_context,
                graph_text=graph_text,
                notes=("graph_only_path",),
            )
            return await self._maybe_synthesize(result, synthesize)

        if decision.route == QueryRoute.RAG_ONLY:
            rag_result = await self._rag_query(query, decision)
            result = DispatchResult(
                query=query,
                decision=decision,
                rag_result=rag_result,
                notes=("rag_only_path",),
            )
            return await self._maybe_synthesize(result, synthesize)

        # HYBRID — run both, in parallel.
        graph_context, rag_result = await asyncio.gather(
            self._graph_query(decision),
            self._rag_query(query, decision),
        )
        graph_text = self._context_builder.build_text_context(graph_context)
        result = DispatchResult(
            query=query,
            decision=decision,
            graph_context=graph_context,
            graph_text=graph_text,
            rag_result=rag_result,
            notes=("hybrid_path",),
        )
        return await self._maybe_synthesize(result, synthesize)

    async def _maybe_synthesize(
        self,
        result: DispatchResult,
        synthesize: bool,
    ) -> DispatchResult:
        if not synthesize or self._synthesizer is None:
            return result
        synthesis = await self._synthesizer.synthesize(result)
        return DispatchResult(
            query=result.query,
            decision=result.decision,
            graph_context=result.graph_context,
            graph_text=result.graph_text,
            rag_result=result.rag_result,
            synthesis=synthesis,
            notes=result.notes,
        )

    # ----------------------------------------------------------------- helpers

    async def _graph_query(self, decision: QueryRoutingDecision) -> GraphReasoningContext:
        """Re-run subgraph extraction for the dispatcher (separate from routing)."""
        if self._graph_extractor is None:
            # No driver — return an empty context. Dispatcher still
            # produces a structured result so the LLM-synthesis layer
            # can decide what to do.
            return GraphReasoningContext(seed_entities=())

        seeds: tuple[str, ...] = tuple(decision.graph_coverage.seed_entities) if decision.graph_coverage else ()
        if not seeds and decision.query_analysis is not None:
            from routing.query_entity_extractor import merge_seed_names
            from synonym_normalization.canonicalizer import canonicalize_text

            seeds = tuple(
                dict.fromkeys(
                    canonicalize_text(ent.canonical or ent.surface)
                    for ent in decision.query_analysis.seed_entities
                )
            )
            seeds = tuple(s for s in seeds if s)

        if not seeds:
            return GraphReasoningContext(seed_entities=())

        try:
            context = await self._graph_extractor.extract_subgraph(
                seed_entity_names=seeds,
                max_hops=self._max_hops,
                limit=self._max_paths,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "Dispatcher graph extraction failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return GraphReasoningContext(seed_entities=seeds)

        # Enrich with contradiction / gap detection.
        try:
            self._reasoner.enrich_context(context)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("GraphReasoner.enrich_context failed: %s: %s", type(exc).__name__, exc)
        return context

    async def _rag_query(
        self,
        query: str,
        decision: QueryRoutingDecision,
    ) -> RAGResult:
        """Call the RAG backend with router-derived filters."""
        entity_filter, numeric_filter = self._build_rag_filters(decision)
        try:
            return await self._rag_client.retrieve(
                query,
                entity_filter=entity_filter,
                numeric_filter=numeric_filter,
                max_results=self._rag_max_results,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("RAG client failed: %s: %s", type(exc).__name__, exc)
            return RAGResult(
                query=query,
                documents=[],
                notes=(f"rag_error:{type(exc).__name__}", str(exc)[:200]),
            )

    @staticmethod
    def _build_rag_filters(
        decision: QueryRoutingDecision,
    ) -> tuple[Sequence[str] | None, Any]:
        """Convert router signals into RAG filter parameters.

        Returns ``(entity_filter, numeric_filter)``. ``entity_filter`` is a
        deduplicated list of canonical seed names; ``numeric_filter`` is a
        ``NumericFilter`` if the query has a numeric constraint.
        """
        seeds: list[str] = []
        if decision.query_analysis is not None:
            from synonym_normalization.canonicalizer import canonicalize_text
            for ent in decision.query_analysis.seed_entities:
                key = canonicalize_text(ent.canonical or ent.surface)
                if key and key not in seeds:
                    seeds.append(key)

        numeric_filter = None
        if (
            decision.query_analysis is not None
            and decision.query_analysis.has_numeric_constraint
        ):
            analysis = decision.query_analysis
            # Surface the parsed numeric bounds to the RAG backend so it can
            # actually filter by them. Previously this block always emitted
            # ``NumericFilter(operator="range", min=None, max=None)`` which
            # was a no-op for every numeric query.
            from .rag_client import NumericFilter
            numeric_filter = NumericFilter(
                property_name=analysis.numeric_unit or "any",
                operator=analysis.numeric_operator or "range",
                min_value=analysis.numeric_min,
                max_value=analysis.numeric_max,
            )

        return (seeds or None), numeric_filter

    @staticmethod
    def extract_query_filters(
        decision: QueryRoutingDecision,
    ) -> dict[str, Any]:
        """Public helper: extract structured filters from a routing decision.

        Returns a dict with keys ``geography``, ``year_min``, ``year_max``,
        ``numeric_min``, ``numeric_max``, ``numeric_unit``. The dispatcher
        applies these via the Cypher helpers; the API exposes them to
        the frontend.
        """
        filters: dict[str, Any] = {}
        analysis = decision.query_analysis
        if analysis is None:
            return filters
        if analysis.has_geo_marker:
            filters["geography"] = "Russia"  # default; overridden below
            # Heuristic: if the query mentions "Россия/отечественная", prefer Russia;
            # if it mentions "зарубежом/мировая", prefer Worldwide.
            normalized = analysis.normalized_query
            if any(w in normalized for w in (
                "росси", "отечеств", "рф", "норильск", "норникел"
            )):
                filters["geography"] = "Russia"
            elif any(w in normalized for w in (
                "зарубеж", "миров", "worldwide", "abroad"
            )):
                filters["geography"] = "Worldwide"
        if analysis.has_temporal_marker:
            # Without a real parser, default to "last 5 years from now" if no
            # explicit year is found.
            import datetime
            now = datetime.date.today().year
            filters["year_min"] = now - 5
            filters["year_max"] = now
        if analysis.has_numeric_constraint:
            # We can't easily parse out the actual range without a
            # full numeric-constraint extractor, so we just record
            # that one is present. Callers can apply their own
            # post-filtering using ``numeric_min`` / ``numeric_max``.
            filters["numeric_present"] = True
        return filters
