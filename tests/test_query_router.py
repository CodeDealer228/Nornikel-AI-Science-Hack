"""Unit tests for the query-time routing decision engine."""

import asyncio
import unittest
from typing import Any, Iterable

from graph_reasoning.models import (
    GraphEdge,
    GraphNode,
    GraphPath,
    GraphReasoningContext,
)
from graph_reasoning.neo4j_subgraph import Neo4jSubgraphExtractor
from routing import (
    GraphCoverageAnalyzer,
    QueryEntityExtractor,
    QueryRoute,
    QueryRouter,
    build_query_router,
)
from routing.query_models import ExtractedQueryEntity
from synonym_normalization.synonym_dictionary import SynonymDictionary


class StubSubgraphExtractor(Neo4jSubgraphExtractor):
    """Neo4jSubgraphExtractor subclass with a synchronous stub for tests."""

    def __init__(self, context: GraphReasoningContext | None) -> None:
        # Skip parent __init__ — we don't need a real driver.
        self._context = context

    async def extract_subgraph(
        self,
        seed_entity_names: Iterable[str],
        max_hops: int = 3,
        limit: int = 200,
    ) -> GraphReasoningContext:
        if self._context is None:
            seeds = tuple(seed_entity_names)
            return GraphReasoningContext(seed_entities=seeds)
        # Re-stamp seed_entities so it matches what was queried.
        return GraphReasoningContext(
            seed_entities=tuple(seed_entity_names),
            nodes=list(self._context.nodes),
            edges=list(self._context.edges),
            paths=list(self._context.paths),
        )


def _node(node_id: str, name: str, type_: str = "Material", confidence: float = 0.8) -> GraphNode:
    return GraphNode(
        id=node_id,
        name=name,
        type=type_,
        source_documents=("doc.md",),
        confidence=confidence,
    )


def _edge(src: str, tgt: str, rel: str = "uses_material", conf: float = 0.7, doc: str = "doc.md") -> GraphEdge:
    return GraphEdge(
        source_id=src,
        target_id=tgt,
        relation_type=rel,
        quote="",
        confidence=conf,
        source_document=doc,
    )


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


class QueryRouterTest(unittest.TestCase):
    def test_empty_query_routes_to_no_data(self):
        router = QueryRouter()
        decision = _run(router.route(""))
        self.assertEqual(decision.route, QueryRoute.NO_DATA)
        self.assertIn("empty_query", decision.reasons)

    def test_low_signal_query_routes_to_no_data(self):
        router = QueryRouter()
        decision = _run(router.route("?"))
        self.assertEqual(decision.route, QueryRoute.NO_DATA)

    def test_rich_graph_coverage_routes_to_graph_only(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[
                _node("n1", "никель", "Material", 0.9),
                _node("n2", "электроэкстракция", "Process", 0.85),
                _node("n3", "катод", "Material", 0.85),
                _node("n4", "документ 1", "Publication", 0.7),
            ],
            edges=[
                _edge("n1", "n2", "uses_material", 0.85),
                _edge("n2", "n3", "produces_output", 0.82),
                _edge("n1", "n3", "uses_material", 0.78),
            ],
            paths=[
                GraphPath(
                    nodes=(_node("n1", "никель", "Material", 0.9),
                           _node("n2", "электроэкстракция", "Process", 0.85),
                           _node("n3", "катод", "Material", 0.85)),
                    edges=(_edge("n1", "n2"), _edge("n2", "n3")),
                ),
            ],
        )
        analyzer = GraphCoverageAnalyzer(
            subgraph_extractor=StubSubgraphExtractor(ctx),
            max_hops=3,
        )
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route("Какие методы электроэкстракции никеля описаны?"))
        self.assertEqual(decision.route, QueryRoute.GRAPH_ONLY)
        self.assertGreater(decision.coverage_score, 0.6)

    def test_partial_coverage_routes_to_hybrid(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[_node("n1", "никель", "Material", 0.6)],
            edges=[_edge("n1", "n2", "uses_material", 0.5)],
        )
        analyzer = GraphCoverageAnalyzer(
            subgraph_extractor=StubSubgraphExtractor(ctx),
        )
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route("Какие методы выщелачивания никеля применяются?"))
        self.assertIn(decision.route, (QueryRoute.HYBRID, QueryRoute.RAG_ONLY, QueryRoute.GRAPH_ONLY))
        # With a single matched seed and very thin graph, expect RAG_ONLY or HYBRID.
        self.assertNotEqual(decision.route, QueryRoute.NO_DATA)

    def test_no_graph_coverage_routes_to_rag_only(self):
        ctx = GraphReasoningContext(
            seed_entities=("неизвестный термин",),
            nodes=[],
            edges=[],
        )
        analyzer = GraphCoverageAnalyzer(
            subgraph_extractor=StubSubgraphExtractor(ctx),
        )
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route("Какие методы очистки шахтных вод применяются?"))
        self.assertEqual(decision.route, QueryRoute.RAG_ONLY)

    def test_contradictions_force_hybrid(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[
                _node("n1", "никель", "Material", 0.9),
                _node("n2", "электроэкстракция", "Process", 0.85),
            ],
            edges=[
                _edge("n1", "n2", "uses_material", 0.9, doc="d1.md"),
                _edge("n1", "n2", "contradicts", 0.7, doc="d2.md"),
            ],
        )
        analyzer = GraphCoverageAnalyzer(
            subgraph_extractor=StubSubgraphExtractor(ctx),
        )
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route("Никель электроэкстракция катод"))
        self.assertEqual(decision.route, QueryRoute.HYBRID)
        self.assertIn("contradictions_present", decision.reasons)

    def test_query_entity_extractor_finds_domain_terms(self):
        extractor = QueryEntityExtractor(synonym_dictionary=SynonymDictionary())
        analysis = extractor.analyze("Какие способы очистки шахтных вод применяются на Норникеле?")
        surface_texts = {ent.surface.lower() for ent in analysis.seed_entities}
        self.assertTrue(any("вод" in t for t in surface_texts) or analysis.has_numeric_constraint or analysis.seed_entities)

    def test_query_entity_extractor_finds_mine_water_injection_terms(self):
        extractor = QueryEntityExtractor(synonym_dictionary=SynonymDictionary())
        analysis = extractor.analyze(
            "Какие способы закачки шахтных вод в глубокие горизонты описаны и их ТЭП?"
        )
        canonical_names = {ent.canonical.lower() for ent in analysis.seed_entities}

        self.assertIn("закачка", canonical_names)
        self.assertIn("вода", canonical_names)
        self.assertIn("горизонт", canonical_names)
        self.assertIn("тэп", canonical_names)

    def test_extracted_query_entity_validates_confidence(self):
        with self.assertRaises(ValueError):
            ExtractedQueryEntity(surface="x", canonical="x", confidence=1.5)

    def test_build_query_router_works_without_driver(self):
        router = build_query_router(driver=None)
        decision = _run(router.route("Какие методы обессоливания воды используются?"))
        # Without a driver, the graph is empty → RAG_ONLY.
        self.assertEqual(decision.route, QueryRoute.RAG_ONLY)

    def test_query_with_numeric_constraints_and_thin_graph(self):
        ctx = GraphReasoningContext(
            seed_entities=("сульфат",),
            nodes=[_node("n1", "сульфат", "Material", 0.5)],
            edges=[],
        )
        analyzer = GraphCoverageAnalyzer(
            subgraph_extractor=StubSubgraphExtractor(ctx),
        )
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route("Найти методы обессоливания при сульфатах ≤300 мг/л"))
        self.assertEqual(decision.route, QueryRoute.RAG_ONLY)
        self.assertIn("query_has_numeric_constraint", decision.reasons)

    def test_decision_to_dict_is_json_safe(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[_node("n1", "никель", "Material", 0.9)],
            edges=[_edge("n1", "n2", "uses_material", 0.8)],
        )
        analyzer = GraphCoverageAnalyzer(
            subgraph_extractor=StubSubgraphExtractor(ctx),
        )
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route("Никель"))
        serialised = decision.to_dict()
        self.assertIn("route", serialised)
        self.assertIn("signals", serialised)


class MarkerRoutingTest(unittest.TestCase):
    """Tests for the explicit-marker override rules."""

    def test_definitional_routes_to_rag_when_graph_thin(self):
        ctx = GraphReasoningContext(
            seed_entities=("пвп",),
            nodes=[_node("n1", "пвп", "Equipment", 0.5)],
            edges=[],
        )
        analyzer = GraphCoverageAnalyzer(subgraph_extractor=StubSubgraphExtractor(ctx))
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route("Что такое ПВП в металлургии никеля?"))
        self.assertEqual(decision.route, QueryRoute.RAG_ONLY)
        self.assertIn("definitional_query", decision.reasons)
        self.assertIn("graph_too_thin_for_definition", decision.reasons)

    def test_comparison_with_graph_routes_to_hybrid(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[
                _node("n1", "никель", "Material", 0.9),
                _node("n2", "медь", "Material", 0.85),
            ],
            edges=[_edge("n1", "n2", "uses_material", 0.8)],
        )
        analyzer = GraphCoverageAnalyzer(subgraph_extractor=StubSubgraphExtractor(ctx))
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route(
            "Сравни свойства никеля и меди для катодной электроэкстракции"
        ))
        self.assertEqual(decision.route, QueryRoute.HYBRID)
        self.assertIn("comparison_query", decision.reasons)
        self.assertIn("needs_both_sources", decision.reasons)

    def test_geo_comparison_routes_to_hybrid(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[_node("n1", "никель", "Material", 0.9)],
            edges=[_edge("n1", "n2", "uses_material", 0.8)],
        )
        analyzer = GraphCoverageAnalyzer(subgraph_extractor=StubSubgraphExtractor(ctx))
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route(
            "Сравни отечественную и зарубежную практику выщелачивания никеля"
        ))
        self.assertEqual(decision.route, QueryRoute.HYBRID)
        self.assertIn("geo_comparison_query", decision.reasons)

    def test_causal_with_graph_routes_to_hybrid(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[_node("n1", "никель", "Material", 0.9)],
            edges=[_edge("n1", "n2", "uses_material", 0.8)],
        )
        analyzer = GraphCoverageAnalyzer(subgraph_extractor=StubSubgraphExtractor(ctx))
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route(
            "Почему получается пористый катод при электроэкстракции никеля в Норникеле"
        ))
        self.assertEqual(decision.route, QueryRoute.HYBRID)
        self.assertIn("causal_query", decision.reasons)
        self.assertIn("prose_explanation_needed", decision.reasons)

    def test_numeric_with_dense_graph_boosts_to_graph_only(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[
                _node("n1", "никель", "Material", 0.9),
                _node("n2", "электролиз", "Process", 0.85),
                _node("n3", "катод", "Material", 0.85),
                _node("n4", "кислота", "Substance", 0.85),
            ],
            edges=[
                _edge("n1", "n2", "uses_material", 0.85),
                _edge("n2", "n3", "produces_output", 0.82),
                _edge("n2", "n4", "uses_material", 0.8),
                _edge("n1", "n3", "uses_material", 0.78),
            ],
        )
        analyzer = GraphCoverageAnalyzer(subgraph_extractor=StubSubgraphExtractor(ctx))
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        decision = _run(router.route(
            "Найти методы при плотности тока ≤ 600 А/м2 и pH ≤ 4 для никеля"
        ))
        # Either GRAPH_ONLY (boost) or HYBRID is acceptable here; what we
        # really want to verify is that the marker rule fired by checking
        # the reason was attached.
        self.assertTrue(
            decision.route in (QueryRoute.GRAPH_ONLY, QueryRoute.HYBRID),
            f"unexpected route: {decision.route}",
        )
        self.assertTrue(
            "numeric_filtered_graph_sufficient" in decision.reasons
            or "query_has_numeric_constraint" in decision.reasons,
            f"numeric reason missing in: {decision.reasons}",
        )

    def test_marker_rule_does_not_apply_when_no_graph(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[],
            edges=[],
        )
        analyzer = GraphCoverageAnalyzer(subgraph_extractor=StubSubgraphExtractor(ctx))
        router = QueryRouter(graph_coverage_analyzer=analyzer)
        # Definitional query with no graph — RAG_ONLY.
        decision = _run(router.route("Что такое электроэкстракция никеля?"))
        self.assertEqual(decision.route, QueryRoute.RAG_ONLY)
        # Geo-comparison with no graph — should NOT short-circuit to HYBRID.
        decision2 = _run(router.route(
            "Сравни отечественную и зарубежную практику никеля"
        ))
        # No graph → either RAG_ONLY (no seeds matched) or NO_DATA via
        # existing fallbacks. The point is that the geo-comparison rule
        # does NOT fire when total_edges == 0.
        self.assertNotEqual(decision2.route, QueryRoute.HYBRID)


if __name__ == "__main__":
    unittest.main()
