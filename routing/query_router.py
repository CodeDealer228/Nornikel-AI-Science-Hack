"""
Query-time routing decision engine.

``QueryRouter`` is the final intelligence layer of the Knowledge
Graph system. It receives a natural-language query, evaluates the
graph coverage (through ``GraphCoverageAnalyzer``), the structural
properties of the query itself (through
``QueryEntityExtractor``), and emits a routing decision:

* ``GRAPH_ONLY``  - graph is sufficient, no document retrieval needed
* ``RAG_ONLY``    - graph has no usable coverage, use document retrieval
* ``HYBRID``      - partial graph coverage: combine graph facts + docs
* ``NO_DATA``     - neither source has any matching information

The router does NOT generate answers. It only emits the decision
and the supporting signals. RAG and answer generation are
external concerns.

Scoring is hybrid: rule-based gating for the extreme cases
(``NO_DATA`` / empty query) and weighted scoring for the rest.
Thresholds are exposed as constructor parameters so they can be
calibrated per deployment.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from graph_reasoning.neo4j_subgraph import Neo4jSubgraphExtractor
from graph_reasoning.reasoner import GraphReasoner
from synonym_normalization.synonym_dictionary import SynonymDictionary

from .graph_coverage import GraphCoverageAnalyzer
from .query_entity_extractor import QueryEntityExtractor, merge_seed_names
from .query_models import (
    GraphCoverageReport,
    QueryAnalysis,
    QueryRoute,
    QueryRoutingDecision,
    QuerySignal,
)

log = logging.getLogger(__name__)


class QueryRouter:
    """Decide whether a user query should be answered by graph, RAG, both, or neither."""

    def __init__(
        self,
        graph_coverage_analyzer: GraphCoverageAnalyzer | None = None,
        query_entity_extractor: QueryEntityExtractor | None = None,
        reasoner: GraphReasoner | None = None,
        # Coverage score weights.
        weight_seed_coverage: float = 0.40,
        weight_multi_hop: float = 0.20,
        weight_relation_density: float = 0.20,
        weight_source_diversity: float = 0.10,
        weight_confidence_in_coverage: float = 0.10,
        # Confidence score weights.
        weight_edge_confidence: float = 0.65,
        weight_gap_inverse: float = 0.35,
        # Decision thresholds.
        graph_only_coverage_threshold: float = 0.62,
        graph_only_confidence_threshold: float = 0.55,
        graph_only_ambiguity_threshold: float = 0.70,
        hybrid_min_coverage: float = 0.20,
        max_paths: int = 200,
    ) -> None:
        self._query_entity_extractor = query_entity_extractor or QueryEntityExtractor()
        self._graph_coverage_analyzer = graph_coverage_analyzer or GraphCoverageAnalyzer(
            reasoner=reasoner,
            max_paths=max_paths,
        )

        total = (
            weight_seed_coverage
            + weight_multi_hop
            + weight_relation_density
            + weight_source_diversity
            + weight_confidence_in_coverage
        )
        if total <= 0:
            raise ValueError("Sum of coverage-score weights must be positive")
        self._w_seed = weight_seed_coverage / total
        self._w_multi_hop = weight_multi_hop / total
        self._w_density = weight_relation_density / total
        self._w_source = weight_source_diversity / total
        self._w_conf_in_cov = weight_confidence_in_coverage / total

        conf_total = weight_edge_confidence + weight_gap_inverse
        if conf_total <= 0:
            raise ValueError("Sum of confidence-score weights must be positive")
        self._w_edge_conf = weight_edge_confidence / conf_total
        self._w_gap = weight_gap_inverse / conf_total

        self._graph_only_cov = graph_only_coverage_threshold
        self._graph_only_conf = graph_only_confidence_threshold
        self._graph_only_amb = graph_only_ambiguity_threshold
        self._hybrid_min = hybrid_min_coverage

    # ------------------------------------------------------------------ public

    async def route(self, query: str) -> QueryRoutingDecision:
        """Decide the routing strategy for a user query."""
        if not query or not query.strip():
            return self._no_data_decision(
                query,
                reasons=("empty_query",),
                notes=("empty_query",),
            )

        analysis = self._query_entity_extractor.analyze(query)

        if analysis.is_low_signal and not analysis.has_numeric_constraint:
            return self._no_data_decision(
                query,
                analysis=analysis,
                reasons=("low_signal_query",),
                notes=analysis.notes,
            )

        seed_names = merge_seed_names(analysis.seed_entities)
        coverage = await self._graph_coverage_analyzer.analyze(seed_names)

        return self._decide(query, analysis, coverage)

    async def route_with_known_seeds(
        self,
        query: str,
        seed_names: Sequence[str],
    ) -> QueryRoutingDecision:
        """Decide a route when callers have already extracted seed entities."""
        if not query or not query.strip():
            return self._no_data_decision(
                query,
                reasons=("empty_query",),
                notes=("empty_query",),
            )

        analysis = self._query_entity_extractor.analyze(query)
        seeds = tuple(dict.fromkeys(name for name in seed_names if name))
        coverage = await self._graph_coverage_analyzer.analyze(seeds)
        return self._decide(query, analysis, coverage)

    # ----------------------------------------------------------------- helpers

    def _no_data_decision(
        self,
        query: str,
        reasons: tuple[str, ...] = (),
        notes: tuple[str, ...] = (),
        analysis: QueryAnalysis | None = None,
    ) -> QueryRoutingDecision:
        return QueryRoutingDecision(
            route=QueryRoute.NO_DATA,
            confidence=0.95,
            coverage_score=0.0,
            ambiguity_score=1.0,
            reasons=reasons,
            signals=(
                QuerySignal("query_chars", len(query or "")),
                QuerySignal("is_no_data", True, weight=2.0),
            ),
            query_analysis=analysis,
            graph_coverage=GraphCoverageReport(notes=notes) if notes else None,
        )

    def _decide(
        self,
        query: str,
        analysis: QueryAnalysis,
        coverage: GraphCoverageReport,
    ) -> QueryRoutingDecision:
        reasons: list[str] = []
        signals: list[QuerySignal] = []

        # ---- coverage score --------------------------------------------------
        seed_score = coverage.seed_coverage_ratio
        multi_hop_score = self._multi_hop_score(coverage.max_hop_observed)
        density_score = self._density_score(coverage.relation_density)
        source_score = self._source_diversity_score(coverage.source_diversity)
        conf_in_cov = coverage.avg_edge_confidence

        coverage_score = (
            self._w_seed * seed_score
            + self._w_multi_hop * multi_hop_score
            + self._w_density * density_score
            + self._w_source * source_score
            + self._w_conf_in_cov * conf_in_cov
        )

        if coverage.has_contradictions:
            # 10% penalty for contradictory evidence — still usable, but the
            # agent should be told to surface the conflict.
            coverage_score = max(0.0, coverage_score - 0.10)
            reasons.append("contradictions_present")

        if coverage.has_knowledge_gaps and coverage.gap_count > 0:
            reasons.append("knowledge_gaps_detected")

        # ---- confidence score ------------------------------------------------
        gap_inverse = 1.0 / (1.0 + coverage.gap_count)
        confidence_score = (
            self._w_edge_conf * coverage.avg_edge_confidence
            + self._w_gap * gap_inverse
        )
        if coverage.has_contradictions:
            confidence_score = max(0.0, confidence_score - 0.05)

        # ---- ambiguity score (lower is better) --------------------------------
        ambiguity_score = self._ambiguity_score(analysis, coverage)
        if analysis.is_low_signal:
            reasons.append("low_signal_query")
        if analysis.has_numeric_constraint:
            reasons.append("query_has_numeric_constraint")
        if analysis.has_geo_marker:
            reasons.append("query_has_geo_marker")
        if analysis.has_temporal_marker:
            reasons.append("query_has_temporal_marker")

        # ---- signals for observability ---------------------------------------
        signals.extend([
            QuerySignal("seed_coverage_ratio", round(seed_score, 4)),
            QuerySignal("multi_hop_score", round(multi_hop_score, 4)),
            QuerySignal("density_score", round(density_score, 4)),
            QuerySignal("source_diversity_score", round(source_score, 4)),
            QuerySignal("coverage_score", round(coverage_score, 4)),
            QuerySignal("confidence_score", round(confidence_score, 4)),
            QuerySignal("ambiguity_score", round(ambiguity_score, 4)),
            QuerySignal("graph_total_nodes", coverage.total_nodes),
            QuerySignal("graph_total_edges", coverage.total_edges),
            QuerySignal("graph_max_hop", coverage.max_hop_observed),
            QuerySignal("graph_contradictions", coverage.contradiction_count),
            QuerySignal("graph_gaps", coverage.gap_count),
            QuerySignal("seed_entity_count", len(analysis.seed_entities)),
            QuerySignal("domain_seed_count", analysis.domain_seed_count),
        ])

        # ---- explicit-marker overrides (run before generic scoring rules) ----
        override = self._apply_marker_rules(
            analysis, coverage, coverage_score, confidence_score, ambiguity_score,
        )
        if override is not None:
            route, override_reasons, override_conf = override
            reasons.extend(override_reasons)
            return self._route_decision(
                route,
                coverage_score=coverage_score,
                confidence_score=confidence_score,
                ambiguity_score=ambiguity_score,
                reasons=tuple(reasons),
                signals=tuple(signals),
                analysis=analysis,
                coverage=coverage,
                confidence=override_conf,
            )

        # ---- decision rules ---------------------------------------------------
        if coverage.total_nodes == 0 and coverage.total_edges == 0:
            reasons.append("graph_empty")
            if analysis.seed_entities:
                reasons.append("seeds_not_in_graph")
                return self._route_decision(
                    QueryRoute.RAG_ONLY,
                    coverage_score=coverage_score,
                    confidence_score=confidence_score,
                    ambiguity_score=ambiguity_score,
                    reasons=tuple(reasons),
                    signals=tuple(signals),
                    analysis=analysis,
                    coverage=coverage,
                    confidence=0.78,
                )
            return self._no_data_decision(
                query,
                analysis=analysis,
                reasons=("graph_empty", "no_seed_entities"),
                notes=("no_seed_entities",),
            )

        if coverage.total_edges == 0 and coverage.total_nodes > 0:
            # We matched nodes but they have no outgoing relations.
            # The graph gives us names but no facts — fall back to RAG.
            reasons.append("matched_nodes_without_relations")
            return self._route_decision(
                QueryRoute.RAG_ONLY,
                coverage_score=coverage_score,
                confidence_score=confidence_score,
                ambiguity_score=ambiguity_score,
                reasons=tuple(reasons),
                signals=tuple(signals),
                analysis=analysis,
                coverage=coverage,
                confidence=0.72,
            )

        if coverage.seed_coverage_ratio == 0.0 and coverage.total_edges == 0:
            reasons.append("no_seeds_matched_and_no_edges")
            return self._route_decision(
                QueryRoute.RAG_ONLY,
                coverage_score=coverage_score,
                confidence_score=confidence_score,
                ambiguity_score=ambiguity_score,
                reasons=tuple(reasons),
                signals=tuple(signals),
                analysis=analysis,
                coverage=coverage,
                confidence=0.7,
            )

        if (
            coverage_score >= self._graph_only_cov
            and confidence_score >= self._graph_only_conf
            and ambiguity_score <= self._graph_only_amb
            and not coverage.has_contradictions
        ):
            reasons.append("graph_only_conditions_met")
            return self._route_decision(
                QueryRoute.GRAPH_ONLY,
                coverage_score=coverage_score,
                confidence_score=confidence_score,
                ambiguity_score=ambiguity_score,
                reasons=tuple(reasons),
                signals=tuple(signals),
                analysis=analysis,
                coverage=coverage,
                confidence=0.82,
            )

        if coverage.has_contradictions or coverage_score < self._graph_only_cov:
            reasons.append("partial_graph_coverage")
            if (
                analysis.has_numeric_constraint
                and coverage.total_edges < 3
            ):
                reasons.append("numeric_constraint_under_covered")
            return self._route_decision(
                QueryRoute.HYBRID,
                coverage_score=coverage_score,
                confidence_score=confidence_score,
                ambiguity_score=ambiguity_score,
                reasons=tuple(reasons),
                signals=tuple(signals),
                analysis=analysis,
                coverage=coverage,
                confidence=0.74,
            )

        # Fallback: too weak to be graph-only and below the hybrid floor.
        reasons.append("below_hybrid_floor")
        return self._route_decision(
            QueryRoute.RAG_ONLY,
            coverage_score=coverage_score,
            confidence_score=confidence_score,
            ambiguity_score=ambiguity_score,
            reasons=tuple(reasons),
            signals=tuple(signals),
            analysis=analysis,
            coverage=coverage,
            confidence=0.7,
        )

    @staticmethod
    def _route_decision(
        route: QueryRoute,
        coverage_score: float,
        confidence_score: float,
        ambiguity_score: float,
        reasons: tuple[str, ...],
        signals: tuple[QuerySignal, ...],
        analysis: QueryAnalysis,
        coverage: GraphCoverageReport,
        confidence: float,
    ) -> QueryRoutingDecision:
        return QueryRoutingDecision(
            route=route,
            confidence=round(min(1.0, max(0.0, confidence)), 4),
            coverage_score=round(coverage_score, 4),
            ambiguity_score=round(ambiguity_score, 4),
            reasons=reasons,
            signals=signals,
            query_analysis=analysis,
            graph_coverage=coverage,
        )

    # ---- score shaping ------------------------------------------------------

    @staticmethod
    def _multi_hop_score(max_hop: int) -> float:
        if max_hop <= 0:
            return 0.0
        if max_hop == 1:
            return 0.35
        if max_hop == 2:
            return 0.7
        return 1.0  # 3 or 4 hops

    @staticmethod
    def _density_score(density: float) -> float:
        if density <= 0:
            return 0.0
        return min(1.0, math.log1p(density) / math.log1p(4.0))

    @staticmethod
    def _source_diversity_score(source_diversity: int) -> float:
        if source_diversity <= 0:
            return 0.0
        return min(1.0, math.log1p(source_diversity) / math.log1p(5.0))

    @staticmethod
    def _ambiguity_score(analysis: QueryAnalysis, coverage: GraphCoverageReport) -> float:
        # Short query → high ambiguity.
        if analysis.token_count <= 1:
            token_score = 1.0
        elif analysis.token_count <= 3:
            token_score = 0.7
        elif analysis.token_count <= 6:
            token_score = 0.4
        else:
            token_score = 0.2

        # Entity count: zero entities is ambiguous; very many is also ambiguous.
        n_entities = len(analysis.seed_entities)
        if n_entities == 0:
            entity_score = 0.8
        elif n_entities == 1:
            entity_score = 0.2
        elif n_entities <= 3:
            entity_score = 0.4
        else:
            entity_score = 0.7

        # When graph coverage is rich, ambiguity drops.
        if coverage.total_edges >= 5 and coverage.seed_coverage_ratio >= 0.6:
            graph_clarity = 0.7
        elif coverage.total_edges > 0:
            graph_clarity = 0.4
        else:
            graph_clarity = 0.0

        ambiguity = (
            0.45 * token_score
            + 0.35 * entity_score
            + 0.20 * (1.0 - graph_clarity)
        )
        return min(1.0, max(0.0, ambiguity))

    def _apply_marker_rules(
        self,
        analysis: QueryAnalysis,
        coverage: GraphCoverageReport,
        coverage_score: float,
        confidence_score: float,
        ambiguity_score: float,
    ) -> tuple[QueryRoute, tuple[str, ...], float] | None:
        """Short-circuit the decision when explicit query markers demand it.

        Returns ``None`` to fall through to the scoring-based rules; returns
        ``(route, reasons, confidence)`` to force a specific decision. Rules
        are evaluated in priority order (first match wins). Unused score
        arguments are kept on the signature so future rules can consult them.
        """
        del confidence_score, ambiguity_score  # currently unused; reserved.

        # 1) Definitional + thin graph → RAG_ONLY. Graph stores facts, not
        #    definitions; prose is the natural place for acronym decoding.
        if analysis.is_definitional and coverage.total_edges < 2:
            return (
                QueryRoute.RAG_ONLY,
                ("definitional_query", "graph_too_thin_for_definition"),
                0.74,
            )

        # 2) Geographic comparison → HYBRID (always, when graph has anything).
        if analysis.is_geo_comparison and coverage.total_edges > 0:
            return (
                QueryRoute.HYBRID,
                ("geo_comparison_query", "needs_both_sources"),
                0.78,
            )

        # 3) Generic comparison with any graph context → HYBRID.
        if (
            analysis.is_comparison
            and not analysis.is_geo_comparison
            and coverage.total_edges > 0
        ):
            return (
                QueryRoute.HYBRID,
                ("comparison_query", "needs_both_sources"),
                0.76,
            )

        # 4) Causal ("почему", "как работает") + graph context → HYBRID.
        if analysis.is_causal and coverage.total_edges >= 1:
            return (
                QueryRoute.HYBRID,
                ("causal_query", "prose_explanation_needed"),
                0.75,
            )

        # 5) Numeric constraint + dense graph near the graph-only threshold
        #    → boost to GRAPH_ONLY. RAG cannot filter by numeric ranges, but
        #    the graph can. Only boosts — never downgrades.
        if (
            analysis.has_numeric_constraint
            and coverage.total_edges >= 3
            and coverage_score >= self._graph_only_cov - 0.1
        ):
            return (
                QueryRoute.GRAPH_ONLY,
                ("numeric_filtered_graph_sufficient",),
                0.80,
            )

        return None


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def build_query_router(
    driver: Any | None = None,
    synonym_dictionary: SynonymDictionary | None = None,
    max_hops: int = 4,
    max_paths: int = 200,
    **kwargs: Any,
) -> QueryRouter:
    """Wire the standard extraction/normalization/reasoning stack to a router.

    The ``driver`` argument is the same async-compatible Neo4j driver
    accepted by ``Neo4jSubgraphExtractor``; pass ``None`` to build a
    router that can still analyse query structure but reports
    ``RAG_ONLY`` for graph coverage (useful for offline tests).
    """
    extractor = Neo4jSubgraphExtractor(driver) if driver is not None else None
    reasoner = GraphReasoner()
    coverage_analyzer = GraphCoverageAnalyzer(
        subgraph_extractor=extractor,
        reasoner=reasoner,
        max_hops=max_hops,
        max_paths=max_paths,
    )
    entity_extractor = QueryEntityExtractor(
        synonym_dictionary=synonym_dictionary or SynonymDictionary(),
    )
    return QueryRouter(
        graph_coverage_analyzer=coverage_analyzer,
        query_entity_extractor=entity_extractor,
        reasoner=reasoner,
        **kwargs,
    )
