"""
Graph coverage analyzer.

Runs the existing ``Neo4jSubgraphExtractor`` against the seed
entities extracted from a user query, then aggregates the
resulting ``GraphReasoningContext`` (using the existing
``GraphReasoner`` for contradiction / gap detection) into a
``GraphCoverageReport`` that the routing engine can reason
about.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Iterable

from graph_reasoning.models import GraphEdge, GraphReasoningContext
from graph_reasoning.neo4j_subgraph import Neo4jSubgraphExtractor
from graph_reasoning.reasoner import GraphReasoner
from synonym_normalization.canonicalizer import canonicalize_text

from .query_models import GraphCoverageReport, GraphHopStats

log = logging.getLogger(__name__)


class GraphCoverageAnalyzer:
    """Quantify how well the knowledge graph covers a set of seed entities."""

    def __init__(
        self,
        subgraph_extractor: Neo4jSubgraphExtractor | None = None,
        reasoner: GraphReasoner | None = None,
        max_hops: int = 4,
        max_paths: int = 200,
    ) -> None:
        self._extractor = subgraph_extractor
        self._reasoner = reasoner or GraphReasoner()
        self._max_hops = max(1, min(max_hops, 4))
        self._max_paths = max(1, max_paths)

    async def analyze(self, seed_names: Iterable[str]) -> GraphCoverageReport:
        """Run subgraph extraction and return aggregated coverage statistics."""
        seed_tuple = tuple(dict.fromkeys(name for name in seed_names if name))
        if not seed_tuple:
            return GraphCoverageReport(
                seed_entities=(),
                notes=("no_seeds",),
            )

        if self._extractor is None:
            return GraphCoverageReport(
                seed_entities=seed_tuple,
                notes=("no_subgraph_extractor",),
            )

        try:
            context = await self._extractor.extract_subgraph(
                seed_entity_names=seed_tuple,
                max_hops=self._max_hops,
                limit=self._max_paths,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "Subgraph extraction failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return GraphCoverageReport(
                seed_entities=seed_tuple,
                notes=(f"extraction_error:{type(exc).__name__}",),
            )

        return self._summarise(context, seed_tuple)

    @staticmethod
    def _seed_lookup_keys(seed_names: tuple[str, ...]) -> set[str]:
        return {canonicalize_text(name) for name in seed_names if name}

    def _summarise(
        self,
        context: GraphReasoningContext,
        seed_tuple: tuple[str, ...],
    ) -> GraphCoverageReport:
        enriched = self._reasoner.enrich_context(context)
        nodes = enriched.nodes
        edges = enriched.edges
        paths = enriched.paths

        lookup = self._seed_lookup_keys(seed_tuple)
        matched: list[str] = []
        unmatched: list[str] = []
        for name in seed_tuple:
            if canonicalize_text(name) in lookup and any(
                canonicalize_text(node.name) == canonicalize_text(name)
                for node in nodes
            ):
                matched.append(name)
            else:
                unmatched.append(name)

        node_count = len(nodes)
        edge_count = len(edges)
        relation_density = (edge_count / node_count) if node_count else 0.0
        avg_confidence = (
            sum(edge.confidence for edge in edges) / edge_count
            if edge_count
            else 0.0
        )
        source_documents = {
            edge.source_document
            for edge in edges
            if edge.source_document
        }

        per_hop = self._per_hop_stats(paths, edges)
        max_hop_observed = max(
            (stat.hop for stat in per_hop if stat.has_path),
            default=0,
        )
        multi_hop_available = max_hop_observed >= 2

        seed_coverage_ratio = (
            len(matched) / len(seed_tuple) if seed_tuple else 0.0
        )

        notes: list[str] = []
        if not node_count and not edge_count:
            notes.append("empty_subgraph")
        if not matched:
            notes.append("no_seed_matches")
        if edge_count and not source_documents:
            notes.append("edges_without_source_documents")
        if multi_hop_available and relation_density < 1.0:
            notes.append("multi_hop_sparse_connectivity")

        return GraphCoverageReport(
            seed_entities=seed_tuple,
            matched_seed_names=tuple(matched),
            unmatched_seed_names=tuple(unmatched),
            seed_coverage_ratio=seed_coverage_ratio,
            total_nodes=node_count,
            total_edges=edge_count,
            relation_density=relation_density,
            source_diversity=len(source_documents),
            avg_edge_confidence=avg_confidence,
            max_hop_observed=max_hop_observed,
            multi_hop_available=multi_hop_available,
            has_contradictions=bool(enriched.contradictions),
            has_knowledge_gaps=bool(enriched.gaps),
            gap_count=len(enriched.gaps),
            contradiction_count=len(enriched.contradictions),
            hop_stats=tuple(per_hop),
            notes=tuple(notes),
        )

    @staticmethod
    def _per_hop_stats(
        paths: Iterable[Any],
        edges: Iterable[GraphEdge],
    ) -> list[GraphHopStats]:
        """Bucket edge statistics by traversal hop (path length)."""
        hop_edges: dict[int, set[tuple[str, str, str, str]]] = defaultdict(set)
        hop_nodes: dict[int, set[str]] = defaultdict(set)
        hop_has_path: dict[int, bool] = defaultdict(bool)

        edge_index: dict[tuple[str, str, str, str], GraphEdge] = {}
        for edge in edges:
            edge_index[
                (edge.source_id, edge.target_id, edge.relation_type, edge.quote)
            ] = edge

        for path in paths:
            try:
                edge_count = len(path.edges)
            except AttributeError:
                edge_count = 0
            try:
                node_count = len(path.nodes)
            except AttributeError:
                node_count = 0
            if edge_count == 0:
                continue
            hop = min(max(edge_count, 1), 4)
            hop_has_path[hop] = True
            for edge in path.edges:
                key = (edge.source_id, edge.target_id, edge.relation_type, edge.quote)
                hop_edges[hop].add(key)
            for node in path.nodes:
                hop_nodes[hop].add(node.id)

        stats: list[GraphHopStats] = []
        for hop in sorted(set(hop_has_path) | set(hop_edges) | {1, 2, 3, 4}):
            unique_edges = list(hop_edges.get(hop, set()))
            if not unique_edges and not hop_has_path.get(hop, False):
                continue
            edge_objs = [
                edge_index[key]
                for key in unique_edges
                if key in edge_index
            ]
            avg_conf = (
                sum(e.confidence for e in edge_objs) / len(edge_objs)
                if edge_objs
                else 0.0
            )
            docs = {
                e.source_document
                for e in edge_objs
                if e.source_document
            }
            stats.append(GraphHopStats(
                hop=hop,
                nodes=len(hop_nodes.get(hop, set())),
                edges=len(unique_edges),
                avg_confidence=avg_conf,
                unique_documents=len(docs),
                has_path=hop_has_path.get(hop, False) or bool(unique_edges),
            ))

        return stats
