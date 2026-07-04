"""search package init.

Public surface:
- ``EdgeHybridSearcher`` for in-memory hybrid search over graph edges.
- ``edge_to_text`` / ``edges_to_chunks`` for rendering edges as searchable docs.
- ``HybridEdgeRAGClient`` for plugging into the project's RAG slot.
"""
from __future__ import annotations

from .edge_adapter import edges_to_chunks, edge_to_text, simple_tokenize
from .edge_hybrid import EdgeHybridSearcher
from .hybrid_search_rag import HybridEdgeRAGClient

__all__ = [
    "EdgeHybridSearcher",
    "HybridEdgeRAGClient",
    "edge_to_text",
    "edges_to_chunks",
    "simple_tokenize",
]