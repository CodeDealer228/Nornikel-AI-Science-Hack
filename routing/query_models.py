"""
Query-time routing data models.

These types are intentionally distinct from the existing
``routing.models`` (extraction-time routing) module. The
extraction-time router decides *which extractor to run* for a
chunk; this module is the query-time decision layer that
decides *which knowledge source* should answer a user query:

    GRAPH_ONLY  - structured knowledge graph has full coverage
    RAG_ONLY    - only the document corpus (RAG) is likely useful
    HYBRID      - combine graph facts with document context
    NO_DATA     - neither source has matching information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QueryRoute(StrEnum):
    """Decision produced by the query-time routing engine."""

    GRAPH_ONLY = "graph_only"
    RAG_ONLY = "rag_only"
    HYBRID = "hybrid"
    NO_DATA = "no_data"


@dataclass(frozen=True)
class QuerySignal:
    """A single named measurement that feeds into routing logic."""

    name: str
    value: float | int | str | bool
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "weight": self.weight}


@dataclass(frozen=True)
class ExtractedQueryEntity:
    """An entity candidate extracted from a user query."""

    surface: str
    canonical: str
    type_hint: str | None = None  # "PER" | "LOC" | "ORG" | None
    char_start: int = 0
    char_end: int = 0
    confidence: float = 0.0
    source: str = "regex"  # "natasha" | "regex" | "synonym" | "llm"

    def __post_init__(self) -> None:
        if not self.surface:
            raise ValueError("ExtractedQueryEntity.surface must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ExtractedQueryEntity.confidence must be in [0, 1], got {self.confidence!r}"
            )


@dataclass(frozen=True)
class GraphHopStats:
    """Aggregated statistics for a single hop level."""

    hop: int
    nodes: int
    edges: int
    avg_confidence: float
    unique_documents: int
    has_path: bool


@dataclass
class GraphCoverageReport:
    """Result of evaluating the graph for a given query."""

    seed_entities: tuple[str, ...] = ()
    matched_seed_names: tuple[str, ...] = ()
    unmatched_seed_names: tuple[str, ...] = ()

    seed_coverage_ratio: float = 0.0
    total_nodes: int = 0
    total_edges: int = 0
    relation_density: float = 0.0
    source_diversity: int = 0
    avg_edge_confidence: float = 0.0
    max_hop_observed: int = 0
    multi_hop_available: bool = False
    has_contradictions: bool = False
    has_knowledge_gaps: bool = False
    gap_count: int = 0
    contradiction_count: int = 0

    hop_stats: tuple[GraphHopStats, ...] = ()

    notes: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.total_nodes == 0 and self.total_edges == 0


@dataclass
class QueryAnalysis:
    """Lightweight structural analysis of the user query."""

    query: str
    normalized_query: str
    char_length: int = 0
    word_count: int = 0
    token_count: int = 0
    has_question_mark: bool = False
    has_numeric_constraint: bool = False
    has_geo_marker: bool = False
    has_temporal_marker: bool = False
    seed_entities: tuple[ExtractedQueryEntity, ...] = ()
    domain_seed_count: int = 0
    is_low_signal: bool = False
    is_definitional: bool = False
    is_causal: bool = False
    is_comparison: bool = False
    is_geo_comparison: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class QueryRoutingDecision:
    """Final routing decision returned by the query router."""

    route: QueryRoute
    confidence: float
    coverage_score: float
    ambiguity_score: float
    reasons: tuple[str, ...] = ()
    signals: tuple[QuerySignal, ...] = ()
    query_analysis: QueryAnalysis | None = None
    graph_coverage: GraphCoverageReport | None = None

    def is_actionable(self) -> bool:
        """Returns True if any knowledge source can answer the query."""
        return self.route != QueryRoute.NO_DATA

    def recommended_secondary_source(self) -> str | None:
        """For hybrid routing, returns the source that should complement the primary."""
        if self.route == QueryRoute.HYBRID:
            return "rag"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": str(self.route),
            "confidence": round(self.confidence, 4),
            "coverage_score": round(self.coverage_score, 4),
            "ambiguity_score": round(self.ambiguity_score, 4),
            "reasons": list(self.reasons),
            "signals": [s.as_dict() for s in self.signals],
            "graph_coverage": (
                {
                    "seed_coverage_ratio": self.graph_coverage.seed_coverage_ratio,
                    "total_nodes": self.graph_coverage.total_nodes,
                    "total_edges": self.graph_coverage.total_edges,
                    "relation_density": round(self.graph_coverage.relation_density, 4),
                    "max_hop_observed": self.graph_coverage.max_hop_observed,
                    "multi_hop_available": self.graph_coverage.multi_hop_available,
                    "has_contradictions": self.graph_coverage.has_contradictions,
                    "gap_count": self.graph_coverage.gap_count,
                }
                if self.graph_coverage is not None
                else None
            ),
        }
