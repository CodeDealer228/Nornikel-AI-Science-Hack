from __future__ import annotations

from collections import Counter, defaultdict

from .context_builder import GraphContextBuilder
from .models import GraphEdge, GraphGap, GraphReasoningContext


class GraphReasoner:
    def __init__(self, context_builder: GraphContextBuilder | None = None) -> None:
        self.context_builder = context_builder or GraphContextBuilder()

    def enrich_context(self, context: GraphReasoningContext) -> GraphReasoningContext:
        context.contradictions = self.detect_contradictions(context)
        context.gaps = self.detect_knowledge_gaps(context)
        return context

    def build_llm_context(self, context: GraphReasoningContext, max_edges: int = 40) -> str:
        enriched = self.enrich_context(context)
        return self.context_builder.build_text_context(enriched, max_edges=max_edges)

    def detect_contradictions(self, context: GraphReasoningContext) -> list[GraphEdge]:
        explicit = [edge for edge in context.edges if edge.relation_type == "contradicts"]
        conflicts: list[GraphEdge] = list(explicit)

        by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
        edges_by_pair: dict[tuple[str, str], list[GraphEdge]] = defaultdict(list)
        for edge in context.edges:
            pair = tuple(sorted((edge.source_id, edge.target_id)))
            by_pair[pair].add(edge.relation_type)
            edges_by_pair[pair].append(edge)

        opposing = {
            ("has_limitation", "has_expected_result"),
            ("replaced_by", "uses_technology"),
        }
        for pair, relation_types in by_pair.items():
            for left, right in opposing:
                if left in relation_types and right in relation_types:
                    conflicts.extend(edges_by_pair[pair])
        return self._unique_edges(conflicts)

    def detect_knowledge_gaps(self, context: GraphReasoningContext) -> list[GraphGap]:
        gaps: list[GraphGap] = []
        degree = Counter()
        for edge in context.edges:
            degree[edge.source_id] += 1
            degree[edge.target_id] += 1

        for node in context.nodes:
            if degree[node.id] == 0:
                gaps.append(GraphGap(
                    code="isolated_entity",
                    entity_id=node.id,
                    message=f"Entity '{node.name}' has no graph relations.",
                    severity="warning",
                ))
            if node.confidence and node.confidence < 0.45:
                gaps.append(GraphGap(
                    code="low_confidence_entity",
                    entity_id=node.id,
                    message=f"Entity '{node.name}' has low confidence ({node.confidence:.2f}).",
                    severity="warning",
                ))

        if context.seed_entities and not context.nodes:
            gaps.append(GraphGap(
                code="missing_seed_entities",
                message="No graph nodes were found for requested seed entities.",
                severity="error",
            ))

        documents = {edge.source_document for edge in context.edges if edge.source_document}
        if context.edges and len(documents) < 2:
            gaps.append(GraphGap(
                code="single_source_support",
                message="Graph context is supported by fewer than two source documents.",
                severity="warning",
            ))

        return gaps

    def _unique_edges(self, edges: list[GraphEdge]) -> list[GraphEdge]:
        seen = set()
        out = []
        for edge in edges:
            key = (edge.source_id, edge.target_id, edge.relation_type, edge.quote)
            if key not in seen:
                seen.add(key)
                out.append(edge)
        return out
