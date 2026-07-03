"""Tests for the RAG factory plug-in interface."""

import unittest

from agent import (
    RAGClient,
    RAGDocument,
    RAGResult,
    StubRAGClient,
    build_rag_client,
    register_rag_backend,
)


class _FakeRAG(RAGClient):
    """Minimal RAGClient implementation used only by tests."""

    def __init__(self, tag: str = "fake") -> None:
        self.tag = tag

    async def retrieve(
        self,
        query: str,
        *,
        entity_filter=None,
        numeric_filter=None,
        max_results: int = 10,
    ) -> RAGResult:
        return RAGResult(
            query=query,
            documents=[RAGDocument(
                doc_id="d1",
                title=self.tag,
                snippet=f"hit for {query}",
                score=0.5,
                source=self.tag,
            )],
            notes=(f"fake_rag:{self.tag}",),
        )


class RAGFactoryTest(unittest.TestCase):
    def setUp(self):
        # Clear any previously-registered test backends.
        from agent import rag_factory
        rag_factory._REGISTRY.pop("fake_rag", None)
        rag_factory._REGISTRY.pop("failing_rag", None)
        rag_factory._REGISTRY.pop("bad_rag", None)

    def test_default_is_stub(self):
        client = build_rag_client()
        self.assertIsInstance(client, StubRAGClient)

    def test_register_and_build(self):
        register_rag_backend(
            "fake_rag",
            _FakeRAG,
            lambda: {"tag": "unit-test"},
        )
        client = build_rag_client("fake_rag")
        self.assertIsInstance(client, _FakeRAG)
        self.assertEqual(client.tag, "unit-test")

    def test_unknown_backend_falls_back_to_stub(self):
        client = build_rag_client("does_not_exist")
        self.assertIsInstance(client, StubRAGClient)

    def test_kwargs_factory_failure_falls_back_to_stub(self):
        def _bad_kwargs():
            raise RuntimeError("simulated env error")

        register_rag_backend("failing_rag", _FakeRAG, _bad_kwargs)
        client = build_rag_client("failing_rag")
        self.assertIsInstance(client, StubRAGClient)

    def test_construction_failure_falls_back_to_stub(self):
        class _BoomOnInit(RAGClient):
            def __init__(self, **kwargs):
                raise RuntimeError("simulated init failure")

            async def retrieve(self, query, *, entity_filter=None, numeric_filter=None, max_results=10):
                return RAGResult(query=query)

        register_rag_backend("bad_rag", _BoomOnInit, lambda: {})
        client = build_rag_client("bad_rag")
        self.assertIsInstance(client, StubRAGClient)

    def test_register_validates_protocol(self):
        class _NotRAG:
            pass

        with self.assertRaises(TypeError):
            register_rag_backend("not_rag", _NotRAG)

    def test_list_registered_backends(self):
        register_rag_backend("fake_rag", _FakeRAG, lambda: {})
        names = __import__("agent.rag_factory", fromlist=["list_registered_backends"]).list_registered_backends()
        self.assertIn("stub", names)
        self.assertIn("fake_rag", names)


if __name__ == "__main__":
    unittest.main()
