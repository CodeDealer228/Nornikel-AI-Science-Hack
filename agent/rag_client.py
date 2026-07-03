"""
RAG client protocol + a default stub implementation.

The dispatcher accepts any object that satisfies ``RAGClient``.
A real RAG backend (Elasticsearch / Vespa / custom) can be
plugged in by implementing the same interface — no dispatcher
changes required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NumericFilter:
    """A simple numeric predicate passed to RAG to narrow the search."""

    property_name: str
    operator: str  # "<=", ">=", "=", "<", ">", "range"
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class RAGDocument:
    doc_id: str
    title: str
    snippet: str
    score: float = 0.0
    source: str = ""
    matched_entities: tuple[str, ...] = ()


@dataclass
class RAGResult:
    query: str
    documents: list[RAGDocument] = field(default_factory=list)
    notes: tuple[str, ...] = ()


@runtime_checkable
class RAGClient(Protocol):
    """Minimal interface a RAG backend must implement to plug into the dispatcher."""

    async def retrieve(
        self,
        query: str,
        *,
        entity_filter: Sequence[str] | None = None,
        numeric_filter: NumericFilter | None = None,
        max_results: int = 10,
    ) -> RAGResult:
        ...


class StubRAGClient:
    """No-op RAG client. Used until the real RAG backend is integrated.

    Returns an empty result with a marker note so the dispatcher
    can produce useful output (e.g. "RAG not configured") instead
    of crashing.
    """

    def __init__(self, message: str = "RAG backend is not configured yet.") -> None:
        self._message = message

    async def retrieve(
        self,
        query: str,
        *,
        entity_filter: Sequence[str] | None = None,
        numeric_filter: NumericFilter | None = None,
        max_results: int = 10,
    ) -> RAGResult:
        log.info(
            "StubRAGClient.retrieve called (query_chars=%d, entities=%s, "
            "numeric=%s); returning empty result",
            len(query),
            entity_filter,
            numeric_filter,
        )
        return RAGResult(
            query=query,
            documents=[],
            notes=("stub_rag_client", self._message),
        )
