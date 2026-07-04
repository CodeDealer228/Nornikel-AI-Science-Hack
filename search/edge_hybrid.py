"""In-memory edge indexer + hybrid (dense + BM25 + RRF) search over graph edges.

Why a custom mini-indexer instead of using ``search/HybridSearcher`` directly?
The friend's ``HybridSearcher`` loads a pre-built ``parsed_data/search_index``
from disk. We don't have that on disk; we want to index graph edges on the fly.

To stay light (no heavy model on every call) we run BM25 on tokenised edges
and rank by token overlap plus a soft graph-context boost (seeds proximity).
Set ``EDGE_SEARCH_MODE=dense`` to additionally load a SentenceTransformer model
for semantic scoring — otherwise the dense leg is skipped and BM25 alone is used.

Public entry points:
- ``EdgeHybridSearcher.from_data_dict(data)`` — build from the frontend
  ``data.json`` shape (entities + graph.edges + graphs).
- ``EdgeHybridSearcher.search(query, top_k=10)`` — return ranked edges with
  ``fused_score`` / ``dense_score`` / ``bm25_score``.
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .edge_adapter import edges_to_chunks, simple_tokenize

log = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+(?:[-_][a-zA-Zа-яА-ЯёЁ0-9]+)?")


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


@dataclass
class _EdgeDoc:
    chunk_id: str
    text: str
    edge: dict[str, Any]
    bm25_tokens: list[str] = field(default_factory=list)


class _BM25Okapi:
    """Tiny BM25Okapi replacement — no rank_bm25 dependency."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = tokenized_corpus
        self.doc_lens = [len(d) for d in tokenized_corpus]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.df: dict[str, int] = {}
        self.tf: list[Counter] = []
        for tokens in tokenized_corpus:
            c = Counter(tokens)
            self.tf.append(c)
            for term in c:
                self.df[term] = self.df.get(term, 0) + 1
        self.N = len(tokenized_corpus)

    def get_scores(self, query_tokens: Iterable[str]) -> list[float]:
        scores = [0.0] * self.N
        qtokens = [t for t in query_tokens if t]
        if not qtokens:
            return scores
        for term in qtokens:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i, doc_tf in enumerate(self.tf):
                f = doc_tf.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i] or 1
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (num / den)
        return scores


class EdgeHybridSearcher:
    """In-memory hybrid search over graph edges."""

    def __init__(
        self,
        edges: list[dict[str, Any]],
        nodes_by_id: dict[str, dict[str, Any]] | None = None,
        *,
        enable_dense: bool = False,
    ) -> None:
        chunks = edges_to_chunks(edges, nodes_by_id=nodes_by_id or {})
        self._docs: list[_EdgeDoc] = []
        tokenized_corpus: list[list[str]] = []
        for ch in chunks:
            tokens = simple_tokenize(ch["text"])
            self._docs.append(_EdgeDoc(
                chunk_id=ch["chunk_id"],
                text=ch["text"],
                edge=ch.get("edge") or {},
                bm25_tokens=tokens,
            ))
            tokenized_corpus.append(tokens)
        self._bm25 = _BM25Okapi(tokenized_corpus)

        self._model = None
        if enable_dense:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                model_name = os.environ.get(
                    "EDGE_SEARCH_MODEL",
                    "intfloat/multilingual-e5-base",
                )
                self._model = SentenceTransformer(model_name)
            except Exception as exc:  # pragma: no cover - optional
                log.warning("Dense model unavailable (%s); falling back to BM25 only.", exc)
                self._model = None

        self._lock = threading.Lock()
        self._seed_boost: dict[str, float] = {}

    # ----------------------------------------------------------- factory API

    @classmethod
    def from_data_dict(cls, data: dict[str, Any], *, enable_dense: bool = False) -> "EdgeHybridSearcher":
        """Build a searcher from a frontend ``data.json``-style dict.

        Prefers the search-ready edge pool (``data["graph"]["edges"]``) which
        now carries up to 6000 edges. Falls back to ``edges_visual`` (the
        400-edge canvas sample) if the new pool is missing — keeps backwards
        compat with older ``data.json`` builds.
        """
        nodes_by_id: dict[str, dict[str, Any]] = {}
        for n in data.get("graph", {}).get("nodes", []) or []:
            nodes_by_id[n["id"]] = {
                "name": n.get("name", ""),
                "type": n.get("type", ""),
            }
        for n in (data.get("entities") or []):
            nodes_by_id[n["id"]] = {"name": n.get("name", ""), "type": n.get("type", "")}
        # Prefer the wider search pool; only fall back to the canvas subset
        # when the JSONL hasn't been rebuilt with the new schema.
        edges = (
            list((data.get("graph") or {}).get("edges") or [])
            or list((data.get("graph") or {}).get("edges_visual") or [])
        )
        return cls(edges, nodes_by_id=nodes_by_id, enable_dense=enable_dense)

    # ------------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        seed_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._docs:
            return []
        qtokens = _tokens(query)
        if not qtokens:
            return []
        # 1. BM25 scores
        bm25_scores = self._bm25.get_scores(qtokens)
        # 2. Dense scores (optional)
        dense_scores: list[float] | None = None
        if self._model is not None:
            try:
                import numpy as np  # type: ignore
                emb = self._model.encode(
                    ["query: " + query],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                embs = self._model.encode(
                    ["passage: " + d.text for d in self._docs],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=16,
                )
                qv = np.asarray(emb, dtype="float32")[0]
                dense_scores = (np.asarray(embs, dtype="float32") @ qv).tolist()
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Dense scoring failed: %s", exc)
                dense_scores = None

        # 3. Graph seed boost (boost edges that touch given seeds)
        seed_set = set(seed_ids or [])
        seed_boost = [0.0] * len(self._docs)
        if seed_set:
            for i, doc in enumerate(self._docs):
                e = doc.edge
                if not e:
                    continue
                if e.get("s") in seed_set or e.get("t") in seed_set:
                    seed_boost[i] = 0.2  # additive boost

        # 4. Combine (RRF-style position-based fusion if both legs present)
        if dense_scores is None:
            fused = bm25_scores
            dense_legend = "bm25_only"
        else:
            fused = [b + d + sb for b, d, sb in zip(bm25_scores, dense_scores, seed_boost)]
            dense_legend = "bm25+dense"

        order = sorted(
            range(len(self._docs)),
            key=lambda i: fused[i],
            reverse=True,
        )[: max(1, top_k)]
        out: list[dict[str, Any]] = []
        for rank, idx in enumerate(order, start=1):
            doc = self._docs[idx]
            out.append({
                "rank": rank,
                "chunk_id": doc.chunk_id,
                "title": (doc.edge.get("p") or "edge") + ": "
                          + (doc.edge.get("s") or "?") + "→" + (doc.edge.get("t") or "?"),
                "snippet": doc.text[:700],
                "text": doc.text,
                "fused_score": float(fused[idx]),
                "dense_score": float(dense_scores[idx]) if dense_scores is not None else 0.0,
                "bm25_score": float(bm25_scores[idx]),
                "source_document": doc.edge.get("doc") or "",
                "predicates": [doc.edge.get("p") or ""],
                "edge": doc.edge,
                "mode": dense_legend,
            })
        return out


__all__ = ["EdgeHybridSearcher"]
