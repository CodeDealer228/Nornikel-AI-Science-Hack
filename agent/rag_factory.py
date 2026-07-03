"""RAG client factory — pluggable slot for the teammate's RAG backend.

The project does not ship a built-in RAG. Instead, this factory
exposes a clean plug-in interface so the team's RAG module can be
attached without changes elsewhere.

There are two ways to plug in a backend:

1. **Entry point** (recommended for installed packages)::

       # setup.cfg / pyproject.toml of the RAG package
       [project.entry-points."kg.rag_backends"]
       elasticsearch = "kg_rag_elasticsearch:ElasticsearchRAGClient"

   The factory will look for an entry point group named
   ``kg.rag_backends`` and instantiate the class with no
   arguments. The class must implement the ``RAGClient`` protocol.

2. **Environment variable** (recommended for ad-hoc registration)::

       RAG_BACKEND=elasticsearch  RAG_ES_URL=http://... python -m agent.cli "..."

   The factory will look up the registered backend by name and
   instantiate it with kwargs read from the environment. Use
   ``register_rag_backend("name", class_, env_kwargs=lambda: {...})``
   in code to wire one up before calling ``build_rag_client()``.

3. **Stub fallback**: if no backend is configured, the factory
   returns a ``StubRAGClient`` that produces empty results with a
   marker note.

Example
-------
::

    # in the RAG package's __init__.py or a startup hook
    from agent import register_rag_backend, RAGClient, RAGResult, RAGDocument

    class ElasticsearchRAGClient:
        def __init__(self, url: str, index: str):
            self._client = ...
        async def retrieve(self, query, *, entity_filter=None, numeric_filter=None, max_results=10):
            # ... call Elasticsearch ...
            return RAGResult(query=query, documents=[RAGDocument(...)])

    def _es_kwargs():
        import os
        return {"url": os.environ["RAG_ELASTICSEARCH_URL"], "index": os.environ.get("RAG_ELASTICSEARCH_INDEX", "kg_chunks")}

    register_rag_backend("elasticsearch", ElasticsearchRAGClient, _es_kwargs)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from .rag_client import RAGClient, StubRAGClient

log = logging.getLogger(__name__)


# A registry of (class, kwargs_factory) tuples keyed by backend name.
_REGISTRY: dict[str, tuple[type, Callable[[], dict[str, Any]]]] = {
    "stub": (StubRAGClient, lambda: {}),
}


def register_rag_backend(
    name: str,
    backend_class: type,
    env_kwargs: Callable[[], dict[str, Any]] | None = None,
) -> None:
    """Register a RAG backend under ``name``.

    ``env_kwargs`` is called at construction time and should return
    a dict of constructor arguments sourced from environment
    variables. If ``None``, the backend will be constructed with
    no arguments.
    """
    if not issubclass(backend_class, RAGClient) and not hasattr(backend_class, "retrieve"):
        raise TypeError(
            f"{backend_class.__name__} does not implement the RAGClient protocol "
            "(missing async retrieve method)"
        )
    _REGISTRY[name] = (backend_class, env_kwargs or (lambda: {}))
    log.info("Registered RAG backend: %s", name)


def _load_entry_point_backends() -> None:
    """Discover RAG backends installed as entry points in the active environment.

    Safe to call multiple times — entry points are loaded only once.
    """
    if getattr(_load_entry_point_backends, "_done", False):
        return
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return

    eps = entry_points()
    # ``entry_points()`` returns a dict on 3.8+ or a SelectableGroups on 3.10+.
    if hasattr(eps, "select"):
        group = eps.select(group="kg.rag_backends")
    else:
        group = eps.get("kg.rag_backends", [])

    for ep in group:
        try:
            cls = ep.load()
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load RAG entry point %s: %s", ep.name, exc)
            continue
        register_rag_backend(ep.name, cls)

    _load_entry_point_backends._done = True  # type: ignore[attr-defined]


def build_rag_client(backend_name: str | None = None) -> RAGClient:
    """Build the RAG client to use in the dispatcher.

    Resolution order:
        1. ``backend_name`` argument if provided.
        2. ``RAG_BACKEND`` env var.
        3. ``stub`` fallback.

    Entry-point backends are loaded on first call.
    """
    _load_entry_point_backends()

    name = (backend_name or os.environ.get("RAG_BACKEND") or "stub").lower()
    entry = _REGISTRY.get(name)
    if entry is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        log.warning("Unknown RAG backend %r; available: %s. Falling back to stub.", name, available)
        return StubRAGClient()

    cls, kwargs_factory = entry
    try:
        kwargs = kwargs_factory()
    except Exception as exc:  # pragma: no cover
        log.warning("RAG backend %r kwargs factory failed: %s; using stub", name, exc)
        return StubRAGClient()

    try:
        instance = cls(**kwargs)
    except Exception as exc:  # pragma: no cover
        log.warning("RAG backend %r construction failed: %s; using stub", name, exc)
        return StubRAGClient()
    return instance


def list_registered_backends() -> list[str]:
    """Return a sorted list of registered RAG backend names."""
    _load_entry_point_backends()
    return sorted(_REGISTRY.keys())


__all__ = [
    "build_rag_client",
    "list_registered_backends",
    "register_rag_backend",
]
