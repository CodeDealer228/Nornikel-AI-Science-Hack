"""Unit tests for the LLM answer synthesizer (no real API calls)."""

import asyncio
import unittest
from typing import Any

from agent import (
    AnswerSynthesizer,
    DispatchResult,
    RAGDocument,
    RAGResult,
    StubRAGClient,
    SynthesisResult,
)
from graph_reasoning.models import (
    GraphEdge,
    GraphNode,
    GraphReasoningContext,
)
from graph_reasoning.neo4j_subgraph import Neo4jSubgraphExtractor
from llm_pipeline_fewshot.llm_parser import YandexGPTClient, YandexGPTError
from routing import GraphCoverageAnalyzer, QueryRoute, QueryRouter
from routing.query_entity_extractor import QueryEntityExtractor
from routing.query_models import (
    GraphCoverageReport,
    QueryAnalysis,
    QueryRoutingDecision,
    QuerySignal,
)
from synonym_normalization.synonym_dictionary import SynonymDictionary


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_dispatch(
    query: str = "никель электроэкстракция",
    route: QueryRoute = QueryRoute.GRAPH_ONLY,
    graph_text: str = "никель --uses_material--> электроэкстракция",
    rag_docs: list[RAGDocument] | None = None,
    notes: tuple[str, ...] = (),
) -> DispatchResult:
    analysis = QueryAnalysis(
        query=query,
        normalized_query=query,
        char_length=len(query),
        word_count=2,
        token_count=2,
        has_question_mark=False,
        seed_entities=(),
    )
    decision = QueryRoutingDecision(
        route=route,
        confidence=0.8,
        coverage_score=0.7,
        ambiguity_score=0.3,
        reasons=("test",),
        signals=(),
        query_analysis=analysis,
        graph_coverage=GraphCoverageReport(
            seed_entities=("никель",),
            matched_seed_names=("никель",),
            total_nodes=2,
            total_edges=1,
        ),
    )
    return DispatchResult(
        query=query,
        decision=decision,
        graph_text=graph_text,
        rag_result=RAGResult(query=query, documents=rag_docs or []),
        notes=notes,
    )


class SynthesizerFallbackTest(unittest.TestCase):
    def test_no_client_renders_context(self):
        synth = AnswerSynthesizer(client=None)
        result = _run(synth.synthesize(_make_dispatch()))
        self.assertFalse(result.used_llm)
        self.assertIn("никель", result.answer)
        self.assertEqual(result.error, "no_llm_client")

    def test_no_data_route_explains_absence(self):
        synth = AnswerSynthesizer(client=None)
        dispatch = _make_dispatch(
            query="???",
            route=QueryRoute.NO_DATA,
            graph_text=None,
        )
        result = _run(synth.synthesize(dispatch))
        self.assertIn("недостаточно данных", result.answer.lower())

    def test_llm_error_falls_back_to_context(self):
        class _BrokenClient:
            model_uri = "yandexgpt-lite"

            def complete(self, system_prompt, user_prompt, max_tokens=None, temperature=None):
                raise YandexGPTError("simulated network error")

        synth = AnswerSynthesizer(client=_BrokenClient())
        result = _run(synth.synthesize(_make_dispatch()))
        self.assertFalse(result.used_llm)
        self.assertIn("simulated network error", result.error)

    def test_rag_only_path_still_renders(self):
        synth = AnswerSynthesizer(client=None)
        dispatch = _make_dispatch(
            route=QueryRoute.RAG_ONLY,
            graph_text=None,
            rag_docs=[RAGDocument(
                doc_id="d1",
                title="Test",
                snippet="Some snippet about никель.",
                score=0.9,
                source="doc.md",
                matched_entities=("никель",),
            )],
        )
        result = _run(synth.synthesize(dispatch))
        self.assertIn("Some snippet", result.answer)


class SynthesizerPromptTest(unittest.TestCase):
    def test_user_prompt_contains_query_and_context(self):
        synth = AnswerSynthesizer(client=None)
        # Internal method; we just want to check it doesn't crash and
        # produces non-empty output.
        prompt = synth._build_user_prompt(_make_dispatch(graph_text="X --y--> Z"))
        self.assertIn("никель электроэкстракция", prompt)
        self.assertIn("X --y--> Z", prompt)
        self.assertIn("graph_only", prompt)

    def test_truncation_works(self):
        synth = AnswerSynthesizer(client=None, max_context_chars=200)
        long_text = "X" * 1000
        truncated = synth._truncate(long_text)
        self.assertLess(len(truncated), 1000)
        self.assertIn("truncated", truncated)

    def test_truncation_skipped_for_short(self):
        synth = AnswerSynthesizer(client=None, max_context_chars=10_000)
        text = "short"
        self.assertEqual(synth._truncate(text), text)


class EndToEndWithSynthesizerTest(unittest.TestCase):
    """A full integration test: router → dispatcher → synthesizer → answer."""

    def test_query_produces_markdown_with_answer_first(self):
        from graph_reasoning.reasoner import GraphReasoner
        from agent.dispatcher import Dispatcher

        ctx = GraphReasoningContext(
            seed_entities=("никель",),
            nodes=[GraphNode(id="n1", name="никель", type="Material", confidence=0.9)],
            edges=[GraphEdge(
                source_id="n1", target_id="n2", relation_type="uses_material",
                confidence=0.8, source_document="doc.md",
            )],
        )

        class _Stub(Neo4jSubgraphExtractor):
            def __init__(self, c):
                self._c = c
            async def extract_subgraph(self, seed_entity_names, max_hops=3, limit=200):
                return GraphReasoningContext(
                    seed_entities=tuple(seed_entity_names),
                    nodes=list(self._c.nodes),
                    edges=list(self._c.edges),
                )

        extractor = _Stub(ctx)
        analyzer = GraphCoverageAnalyzer(
            subgraph_extractor=extractor, reasoner=GraphReasoner(),
        )
        entity_extractor = QueryEntityExtractor(SynonymDictionary())
        router = QueryRouter(
            graph_coverage_analyzer=analyzer,
            query_entity_extractor=entity_extractor,
            reasoner=GraphReasoner(),
        )
        synth = AnswerSynthesizer(client=None)
        dispatcher = Dispatcher(
            router=router,
            graph_extractor=extractor,
            rag_client=StubRAGClient(),
            synthesizer=synth,
        )
        result = _run(dispatcher.dispatch("никель электроэкстракция катод"))
        self.assertIsNotNone(result.synthesis)
        self.assertFalse(result.synthesis.used_llm)  # no real client
        md = result.to_markdown()
        # The synthesized answer should appear before the dispatch detail.
        self.assertLess(
            md.find("Краткий вывод") if "Краткий вывод" in md else md.find("Dispatch"),
            md.find("Dispatch detail") if "Dispatch detail" in md else len(md),
        )


if __name__ == "__main__":
    unittest.main()
