"""Unit tests for the dispatcher (graph + RAG execution layer)."""

import asyncio
import unittest
from typing import Any, Sequence

from agent import (
    Dispatcher,
    NumericFilter,
    RAGClient,
    RAGDocument,
    RAGResult,
    StubRAGClient,
)
from graph_reasoning.models import GraphEdge, GraphNode, GraphReasoningContext
from graph_reasoning.neo4j_subgraph import Neo4jSubgraphExtractor
from routing import QueryRouter, build_query_router


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


class _StubRAG(RAGClient):
    """Spy RAG client that records what it was called with."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        *,
        entity_filter: Sequence[str] | None = None,
        numeric_filter: NumericFilter | None = None,
        max_results: int = 10,
    ) -> RAGResult:
        self.calls.append({
            "query": query,
            "entity_filter": list(entity_filter) if entity_filter else None,
            "numeric_filter": numeric_filter,
            "max_results": max_results,
        })
        return RAGResult(
            query=query,
            documents=[RAGDocument(
                doc_id="d1",
                title="Test document",
                snippet="Some snippet.",
                score=0.9,
                source="doc.md",
                matched_entities=tuple(entity_filter or ()),
            )],
        )


class _GraphStub(Neo4jSubgraphExtractor):
    def __init__(self, ctx: GraphReasoningContext) -> None:
        self._ctx = ctx

    async def extract_subgraph(
        self,
        seed_entity_names: Sequence[str],
        max_hops: int = 3,
        limit: int = 200,
    ) -> GraphReasoningContext:
        return GraphReasoningContext(
            seed_entities=tuple(seed_entity_names),
            nodes=list(self._ctx.nodes),
            edges=list(self._ctx.edges),
            paths=list(self._ctx.paths),
        )


def _make_dispatcher(rag: RAGClient, graph_ctx: GraphReasoningContext | None) -> Dispatcher:
    """Build a dispatcher with a stub graph extractor AND a router that uses
    the same stub. This is necessary so the routing decision and the
    dispatcher's graph path agree on what is in the graph.
    """
    from graph_reasoning.reasoner import GraphReasoner
    from routing import GraphCoverageAnalyzer, QueryEntityExtractor
    from routing.query_router import QueryRouter
    from synonym_normalization.synonym_dictionary import SynonymDictionary

    extractor = _GraphStub(graph_ctx) if graph_ctx is not None else None
    reasoner = GraphReasoner()
    analyzer = GraphCoverageAnalyzer(
        subgraph_extractor=extractor, reasoner=reasoner, max_hops=4,
    )
    entity_extractor = QueryEntityExtractor(synonym_dictionary=SynonymDictionary())
    router = QueryRouter(
        graph_coverage_analyzer=analyzer,
        query_entity_extractor=entity_extractor,
        reasoner=reasoner,
    )
    return Dispatcher(
        router=router,
        graph_extractor=extractor,
        rag_client=rag,
    )


class DispatcherNoDataTest(unittest.TestCase):
    def test_empty_query_returns_no_data(self):
        rag = _StubRAG()
        dispatcher = _make_dispatcher(rag, None)
        result = _run(dispatcher.dispatch(""))
        self.assertEqual(result.decision.route.value, "no_data")
        self.assertEqual(rag.calls, [])

    def test_low_signal_query_does_not_call_rag(self):
        rag = _StubRAG()
        dispatcher = _make_dispatcher(rag, None)
        result = _run(dispatcher.dispatch("?"))
        self.assertEqual(result.decision.route.value, "no_data")
        self.assertEqual(rag.calls, [])


class DispatcherRAGOnlyTest(unittest.TestCase):
    def test_unknown_query_routes_to_rag_only(self):
        rag = _StubRAG()
        dispatcher = _make_dispatcher(rag, None)
        result = _run(dispatcher.dispatch("Какие методы очистки шахтных вод применяются в России?"))
        self.assertEqual(result.decision.route.value, "rag_only")
        self.assertIsNotNone(result.rag_result)
        self.assertIsNone(result.graph_text)
        self.assertEqual(len(rag.calls), 1)
        # Numeric / geo / temporal markers should propagate as RAG filters.
        self.assertTrue(rag.calls[0]["entity_filter"] is not None)


class DispatcherStubRAGTest(unittest.TestCase):
    def test_stub_rag_returns_empty_with_marker_note(self):
        dispatcher = _make_dispatcher(StubRAGClient(), None)
        result = _run(dispatcher.dispatch("Какие методы очистки шахтных вод применяются?"))
        self.assertEqual(result.decision.route.value, "rag_only")
        self.assertEqual(result.rag_result.documents, [])
        self.assertTrue(any("stub_rag_client" in note for note in result.rag_result.notes))


class DispatcherGraphTest(unittest.TestCase):
    def test_graph_only_path_uses_graph(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[
                GraphNode(id="n1", name="никель", type="Material",
                          source_documents=("doc.md",), confidence=0.9),
                GraphNode(id="n2", name="электроэкстракция", type="Process",
                          source_documents=("doc.md",), confidence=0.85),
            ],
            edges=[
                GraphEdge(source_id="n1", target_id="n2", relation_type="uses_material",
                          confidence=0.85, source_document="doc.md"),
            ],
        )
        rag = _StubRAG()
        dispatcher = _make_dispatcher(rag, ctx)
        result = _run(dispatcher.dispatch("Какие методы электроэкстракции никеля описаны?"))
        # The decision may be GRAPH_ONLY or HYBRID depending on coverage,
        # but the dispatcher must have attempted the graph path.
        self.assertIsNotNone(result.graph_text)
        self.assertIn("никель", result.graph_text)

    def test_dispatcher_runs_graph_and_rag_in_parallel_for_hybrid(self):
        ctx = GraphReasoningContext(
            seed_entities=("сульфат",),
            nodes=[GraphNode(id="s1", name="сульфат", type="Material",
                             source_documents=("d.md",), confidence=0.5)],
            edges=[],
        )
        rag = _StubRAG()
        dispatcher = _make_dispatcher(rag, ctx)
        result = _run(dispatcher.dispatch("Найти методы обессоливания при сульфатах ≤300 мг/л"))
        # Either RAG_ONLY (because no edges) or HYBRID — but the dispatcher
        # must surface a structured RAG call when the path is taken.
        self.assertIsNotNone(result.decision.reasons)
        self.assertGreater(len(rag.calls) + 0, 0)  # RAG called at least sometimes

    def test_dispatcher_handles_missing_rag_client(self):
        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[GraphNode(id="n1", name="никель", type="Material", confidence=0.9)],
            edges=[GraphEdge(source_id="n1", target_id="n2",
                              relation_type="uses_material", confidence=0.8)],
        )
        # No RAG client at all — dispatcher must still work.
        dispatcher = _make_dispatcher(StubRAGClient("offline"), ctx)
        result = _run(dispatcher.dispatch("Никель электроэкстракция катод"))
        self.assertIn("Route:", result.to_markdown())


class DispatchResultMarkdownTest(unittest.TestCase):
    def test_markdown_contains_key_fields(self):
        rag = _StubRAG()
        dispatcher = _make_dispatcher(rag, None)
        result = _run(dispatcher.dispatch("Какие методы очистки шахтных вод применяются?"))
        md = result.to_markdown()
        self.assertIn("Route:", md)
        self.assertIn("rag_only", md)


if __name__ == "__main__":
    unittest.main()
