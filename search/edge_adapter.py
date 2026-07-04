"""Edge adapter: turn graph edges into chunks consumable by ``search.HybridSearcher``.

The friend's search module indexes document chunks (text + provenance). We want
to run the same hybrid (dense + BM25 + RRF) search over **graph edges** —
each edge becomes a small document:

    <source_name> --<predicate>--> <target_name>.
    Quote: «<quote>»
    Source document: <source_document>

This keeps search behaviour consistent with chunk search but ranks edges
instead of text spans.

The adapter does not require a built index — the companion
``EdgeHybridSearcher`` builds an in-memory index on first use, so the RAG
client can answer immediately without precomputing `parsed_data/search_index`.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

log = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+(?:[-_][a-zA-Zа-яА-ЯёЁ0-9]+)?")


def simple_tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def edge_to_text(
    *,
    source_name: str,
    predicate: str,
    target_name: str,
    quote: str = "",
    source_document: str = "",
    source_type: str = "",
    target_type: str = "",
) -> str:
    """Render an edge as a short, search-friendly text blob."""
    parts: list[str] = []
    if source_type:
        parts.append(f"[{source_type}]")
    parts.append(source_name or "")
    parts.append(f"--{predicate}-->")
    if target_type:
        parts.append(f"[{target_type}]")
    parts.append(target_name or "")
    parts.append(".")
    if quote:
        parts.append(f"\nЦитата: «{quote}»")
    if source_document:
        parts.append(f"\nДокумент: {source_document}")
    return " ".join(parts).strip()


def edges_to_chunks(
    edges: Iterable[dict[str, Any]],
    *,
    nodes_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert raw edge dicts into ``chunks.jsonl``-shaped records.

    ``edges`` items may carry: ``s``, ``t``, ``p``, ``quote``, ``doc``.
    Names are resolved via ``nodes_by_id`` when available.
    """
    out: list[dict[str, Any]] = []
    for idx, edge in enumerate(edges):
        s_id = edge.get("s") or edge.get("source_id")
        t_id = edge.get("t") or edge.get("target_id")
        s_node = (nodes_by_id or {}).get(s_id, {}) if s_id else {}
        t_node = (nodes_by_id or {}).get(t_id, {}) if t_id else {}
        s_name = s_node.get("name") or s_id or ""
        t_name = t_node.get("name") or t_id or ""
        s_type = s_node.get("type") or ""
        t_type = t_node.get("type") or ""
        predicate = edge.get("p") or edge.get("relation_type") or "related_to"
        quote = edge.get("quote") or ""
        doc = edge.get("doc") or edge.get("source_document") or ""
        chunk_text = edge_to_text(
            source_name=s_name,
            predicate=predicate,
            target_name=t_name,
            quote=quote,
            source_document=doc,
            source_type=s_type,
            target_type=t_type,
        )
        out.append({
            "chunk_id": f"edge::{idx}::{s_id}::{t_id}",
            "text": chunk_text,
            "provenance": {
                "source_document": doc,
                "heading_path": [f"edge:{predicate}"],
                "char_start": 0,
                "char_end": len(chunk_text),
                "edge_ids": [s_id, t_id],
                "predicate": predicate,
            },
            "edge": {
                "s": s_id,
                "t": t_id,
                "p": predicate,
                "quote": quote,
                "doc": doc,
            },
        })
    return out


__all__ = ["edge_to_text", "edges_to_chunks", "simple_tokenize"]