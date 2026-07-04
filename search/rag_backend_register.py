"""Auto-register the hybrid edge RAG backend on import.

Usage::

    # In api/server.py or wherever the dispatcher is built:
    import search.rag_backend_register  # noqa: F401
    rag = build_rag_client()  # honours RAG_BACKEND env var

Or simply set::

    RAG_BACKEND=hybrid_search  uvicorn api.server:app --port 8080

The backend is idempotent — safe to import multiple times.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    try:
        from agent.rag_factory import register_rag_backend
        from search.hybrid_search_rag import HybridEdgeRAGClient, _kwargs
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not register hybrid_search RAG backend: %s", exc)
        return
    try:
        register_rag_backend("hybrid_search", HybridEdgeRAGClient, _kwargs)
        _registered = True
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("register_rag_backend(hybrid_search) failed: %s", exc)


register()