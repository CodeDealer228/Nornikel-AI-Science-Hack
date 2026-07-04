"""RAG backend that runs hybrid (dense + BM25) search over graph edges.

Adapter from the friend's ``search`` module to the project's ``RAGClient``
protocol defined in ``agent/rag_client.py``.

The searcher is built from the latest frontend ``data.json`` (graph + entities),
so no Neo4j instance is required. If the JSON is missing, the client falls
back to a stub note and returns no documents.

Registration::

    from agent.rag_factory import register_rag_backend
    from search.hybrid_search_rag import HybridEdgeRAGClient, _kwargs
    register_rag_backend("hybrid_search", HybridEdgeRAGClient, _kwargs)

Environment knobs:
    RAG_DATA_JSON          path to frontend/data.json (default: frontend/data.json)
    RAG_EDGE_TOP_K         max results per query (default: 10)
    RAG_ENABLE_DENSE       1 to load SentenceTransformer (default: 0; BM25 only)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

from agent.rag_client import (
    NumericFilter,
    RAGClient,
    RAGDocument,
    RAGResult,
)

from .edge_hybrid import EdgeHybridSearcher

log = logging.getLogger(__name__)


_DEFAULT_DATA_JSON = Path(__file__).resolve().parent.parent / "frontend" / "data.json"


class HybridEdgeRAGClient:
    """RAG backend that ranks graph edges by BM25 (+ optional dense)."""

    def __init__(
        self,
        data_json: str | os.PathLike[str] | None = None,
        top_k: int = 10,
        enable_dense: bool = False,
    ) -> None:
        self._path = Path(data_json) if data_json else _DEFAULT_DATA_JSON
        self._top_k = max(1, int(top_k))
        self._enable_dense = bool(enable_dense)
        self._searcher: EdgeHybridSearcher | None = None
        self._notes: list[str] = []
        self._load()

    # ----------------------------------------------------------- internals

    def _load(self) -> None:
        if not self._path.exists():
            self._notes.append(f"data_missing:{self._path}")
            log.warning("HybridEdgeRAGClient: data.json not found at %s", self._path)
            return
        try:
            import json

            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            self._notes.append(f"data_load_error:{type(exc).__name__}:{exc}")
            log.warning("HybridEdgeRAGClient: failed to load %s: %s", self._path, exc)
            return
        try:
            self._searcher = EdgeHybridSearcher.from_data_dict(
                data, enable_dense=self._enable_dense
            )
            self._notes.append(f"edge_hybrid_ready:{self._path}")
            log.info(
                "HybridEdgeRAGClient: indexed %d edges (dense=%s)",
                len(data.get("graph", {}).get("edges", []) or []),
                self._enable_dense,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._notes.append(f"index_error:{type(exc).__name__}:{exc}")
            log.warning("HybridEdgeRAGClient: index build failed: %s", exc)
            self._searcher = None

    # ------------------------------------------------------------ protocol

    async def retrieve(
        self,
        query: str,
        *,
        entity_filter: Sequence[str] | None = None,
        numeric_filter: NumericFilter | None = None,
        max_results: int = 10,
    ) -> RAGResult:
        if self._searcher is None:
            return RAGResult(
                query=query,
                documents=[],
                notes=tuple(self._notes) + ("edge_hybrid_disabled",),
            )

        results = self._searcher.search(
            query,
            top_k=max_results or self._top_k,
            seed_ids=entity_filter,
        )
        docs: list[RAGDocument] = []
        for hit in results:
            edge = hit.get("edge") or {}
            docs.append(
                RAGDocument(
                    doc_id=hit.get("chunk_id") or "",
                    title=hit.get("title") or "edge",
                    snippet=hit.get("snippet") or hit.get("text") or "",
                    score=float(hit.get("fused_score") or 0.0),
                    source=hit.get("source_document") or "",
                    matched_entities=(
                        (edge.get("s") or "", edge.get("t") or "")
                        if edge else ()
                    ),
                )
            )
        notes: tuple[str, ...] = tuple(self._notes) + (
            f"edge_hybrid_results:{len(docs)}",
        )
        return RAGResult(query=query, documents=docs, notes=notes)


# ------------------------------------------------------------ env factory


def _kwargs() -> dict[str, object]:
    return {
        "data_json": os.environ.get("RAG_DATA_JSON") or str(_DEFAULT_DATA_JSON),
        "top_k": int(os.environ.get("RAG_EDGE_TOP_K", "10")),
        "enable_dense": os.environ.get("RAG_ENABLE_DENSE", "0") == "1",
    }


__all__ = ["HybridEdgeRAGClient", "_kwargs"]