"""FastAPI server exposing the Knowledge Graph as a REST API.

Endpoints:
    POST /query         — full pipeline: routing + dispatch + synthesis
    POST /route         — routing decision only (no graph/RAG execution)
    GET  /health        — liveness probe (always 200 if process is up)
    GET  /ready         — readiness probe (checks Neo4j connectivity)
    GET  /stats         — graph statistics
    GET  /entities      — list entities by name/type/geography/year
    GET  /metrics       — Prometheus-style text metrics (no auth)

Auth (optional):
    Set ``API_KEY`` env var. If non-empty, callers must send
    ``X-API-Key: <value>`` header on protected endpoints.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Sequence

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import (
    AnswerSynthesizer,
    DeepResearchAgent,
    Dispatcher,
    DispatchResult,
    StubRAGClient,
)
from agent.rag_factory import build_rag_client
try:
    # Auto-register hybrid edge RAG backend (BM25 over graph edges).
    import search.rag_backend_register  # noqa: F401
except Exception as exc:  # pragma: no cover - defensive
    log.debug("search.rag_backend_register import failed: %s", exc)
from config import APISettings, get_settings
from graph_reasoning import (
    Neo4jSubgraphExtractor,
    entities_by_geography,
    entities_by_name,
    entities_by_numeric_value,
    entities_by_year_range,
    graph_statistics,
)
from logging_setup import configure_logging, get_logger, set_request_id
from llm_pipeline_fewshot.llm_parser import (
    MockLLMClient,
    create_llm_client,
)
from neo4j_integration.neo4j_config import Neo4jConfig
from neo4j_integration.neo4j_loader import Neo4jLoader
from ontology import EntityType
from routing import build_query_router
from routing.query_models import QueryRoute

log = get_logger(__name__)


# ----------------------------------------------------------------- app state


class AppState:
    """Holds shared resources for the API server."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.driver: Any = None
        self.dispatcher: Dispatcher | None = None
        self.deep_research_agent: DeepResearchAgent | None = None
        self.started_at: float = time.time()
        self.request_count: int = 0
        self.error_count: int = 0
        self.synthesis_calls: int = 0


STATE = AppState()


# ---------------------------------------------------------------- pydantic


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query in natural language.")
    synthesize: bool = Field(default=True, description="Run LLM synthesis (if configured).")
    top_k: int = Field(default=8, ge=1, le=50, description="Max RAG results.")


class QueryResponse(BaseModel):
    query: str
    route: str
    confidence: float
    coverage_score: float
    ambiguity_score: float
    reasons: list[str]
    answer: str
    used_llm: bool
    graph_text: str | None = None
    rag_documents: list[dict[str, Any]] = Field(default_factory=list)
    request_id: str


_DEEP_RESEARCH_MAX_TOOL_CALLS = get_settings().deep_research.max_tool_calls


class DeepResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research query in natural language.")
    max_iterations: int = Field(
        default=_DEEP_RESEARCH_MAX_TOOL_CALLS,
        ge=1,
        le=_DEEP_RESEARCH_MAX_TOOL_CALLS,
        description=(
            "Total graph_search + rag_search tool-call budget for this run "
            "(the agent decides the mix; capped by DEEP_RESEARCH_MAX_TOOL_CALLS)."
        ),
    )


class DeepResearchIterationResponse(BaseModel):
    index: int
    query: str
    route: str
    confidence: float
    coverage_score: float
    graph_text: str
    rag_documents: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DeepResearchResponse(BaseModel):
    query: str
    answer: str
    used_llm: bool
    model_uri: str
    iterations: list[DeepResearchIterationResponse]
    follow_up_queries: list[str]
    sources: list[str]
    notes: list[str]
    request_id: str


class RouteOnlyRequest(BaseModel):
    query: str = Field(..., min_length=1)


class RouteOnlyResponse(BaseModel):
    query: str
    route: str
    confidence: float
    coverage_score: float
    ambiguity_score: float
    reasons: list[str]
    markers: dict[str, bool]
    request_id: str


class StatsResponse(BaseModel):
    total_nodes: int
    total_relationships: int
    by_label: dict[str, int]


# ------------------------------------------------------------------- auth


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if not STATE.settings.api.api_key:
        return  # auth disabled
    if x_api_key != STATE.settings.api.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------- startup/shutdown


async def _connect_neo4j() -> Any | None:
    """Open a Neo4j driver. Returns None on failure (offline mode)."""
    cfg = STATE.settings.neo4j
    try:
        # Use the loader's driver-creation snippet to ensure same version
        # of neo4j package.
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        # Sanity check.
        async with driver.session() as session:
            await session.run("RETURN 1")
        log.info("Connected to Neo4j at %s", cfg.uri)
        return driver
    except Exception as exc:
        log.warning("Neo4j connection failed: %s: %s", type(exc).__name__, exc)
        return None


def _build_dispatcher() -> Dispatcher:
    """Build a Dispatcher with the currently available backends.

    The RAG client is constructed via ``build_rag_client()``, which
    honours the ``RAG_BACKEND`` env var and any registered
    entry-points. Falls back to ``StubRAGClient`` if no backend is
    configured.
    """
    try:
        rag = build_rag_client()
        log.info("RAG client ready: %s", type(rag).__name__)
    except Exception as exc:
        log.warning("RAG client build failed: %s: %s; using stub", type(exc).__name__, exc)
        rag = StubRAGClient()

    synth: AnswerSynthesizer | None = None
    # Synthesis follows LLM_CLIENT_MODE (so `LLM_CLIENT_MODE=deepseek` makes
    # the API synthesize answers via DeepSeek-on-Yandex, same as ingest).
    # MockLLMClient is skipped — it emits a NER/RE JSON fixture, not a
    # user-facing answer, so we fall back to context-only render instead.
    try:
        client = create_llm_client()
        if isinstance(client, MockLLMClient):
            log.info("AnswerSynthesizer skipped (LLM client is mock).")
        else:
            synth = AnswerSynthesizer(client=client)
            log.info("AnswerSynthesizer configured with %s", client.model_uri)
    except Exception as exc:
        log.warning("AnswerSynthesizer setup failed: %s: %s", type(exc).__name__, exc)

    if STATE.driver is None:
        log.info("Building dispatcher in offline mode (no Neo4j).")
        return Dispatcher(
            router=build_query_router(),
            graph_extractor=None,
            rag_client=rag,
            synthesizer=synth,
        )

    extractor = Neo4jSubgraphExtractor(STATE.driver)
    return Dispatcher(
        router=build_query_router(driver=STATE.driver),
        graph_extractor=extractor,
        rag_client=rag,
        synthesizer=synth,
    )


def _build_deep_research_agent(dispatcher: Dispatcher) -> DeepResearchAgent:
    """Build the iterative research agent over the existing dispatcher tools."""
    client: Any | None = None
    try:
        candidate = create_llm_client()
        if isinstance(candidate, MockLLMClient):
            log.info("DeepResearchAgent will use deterministic fallback (LLM client is mock).")
        else:
            client = candidate
            log.info("DeepResearchAgent configured with %s", client.model_uri)
    except Exception as exc:
        log.warning("DeepResearchAgent LLM setup failed: %s: %s", type(exc).__name__, exc)
    return DeepResearchAgent(
        rag_client=dispatcher.rag_client,
        graph_extractor=dispatcher.graph_extractor,
        llm_client=client,
        max_hops=dispatcher.max_hops,
        max_paths=dispatcher.max_paths,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(
        level=STATE.settings.logging.level,
        fmt=STATE.settings.logging.format,
        log_dir=str(STATE.settings.logging.log_dir) if STATE.settings.logging.log_dir else None,
    )
    log.info("Starting Nornikel Knowledge Graph API server.")
    STATE.driver = await _connect_neo4j()
    STATE.dispatcher = _build_dispatcher()
    STATE.deep_research_agent = _build_deep_research_agent(STATE.dispatcher)
    try:
        yield
    finally:
        log.info("Shutting down API server.")
        if STATE.driver is not None:
            try:
                await STATE.driver.close()
            except Exception:  # pragma: no cover
                pass


# ----------------------------------------------------------------- FastAPI


app = FastAPI(
    title="Nornikel Knowledge Graph API",
    version="0.1.0",
    description="KG-RAG hybrid: query-time routing + LLM synthesis.",
    lifespan=lifespan,
)

if get_settings().api.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------- endpoints


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_sec": round(time.time() - STATE.started_at, 2),
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    neo4j_ok = False
    if STATE.driver is not None:
        try:
            async with STATE.driver.session() as session:
                await session.run("RETURN 1")
            neo4j_ok = True
        except Exception as exc:
            log.debug("Neo4j readiness check failed: %s", exc)
    return {
        "ready": neo4j_ok,
        "neo4j_connected": neo4j_ok,
        "synthesis_configured": STATE.dispatcher is not None
        and STATE.dispatcher._synthesizer is not None,  # noqa: SLF001 (introspection)
    }


@app.get("/stats", response_model=StatsResponse, dependencies=[Depends(require_api_key)])
async def stats() -> StatsResponse:
    if STATE.driver is None:
        raise HTTPException(status_code=503, detail="Neo4j not connected")
    counts = await graph_statistics(STATE.driver)
    return StatsResponse(
        total_nodes=counts.pop("total_nodes", 0),
        total_relationships=counts.pop("total_relationships", 0),
        by_label=counts,
    )


@app.get("/entities", dependencies=[Depends(require_api_key)])
async def entities(
    name: list[str] | None = Query(default=None, description="Filter by name (any-of)."),
    geography: str | None = Query(default=None),
    min_year: int | None = Query(default=None, ge=1800, le=2100),
    max_year: int | None = Query(default=None, ge=1800, le=2100),
    property_name: str | None = Query(default=None),
    min_value: float | None = Query(default=None),
    max_value: float | None = Query(default=None),
    unit: str | None = Query(default=None),
    entity_type: str | None = Query(default=None, description="e.g. 'Material', 'Process'."),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    if STATE.driver is None:
        raise HTTPException(status_code=503, detail="Neo4j not connected")
    etype = None
    if entity_type:
        try:
            etype = EntityType(entity_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown entity_type: {entity_type}")

    results: list[dict[str, Any]] = []
    if name:
        results.extend(await entities_by_name(STATE.driver, name, entity_type=etype, limit=limit))
    if geography:
        results.extend(await entities_by_geography(STATE.driver, geography, entity_type=etype, limit=limit))
    if min_year is not None or max_year is not None:
        results.extend(await entities_by_year_range(
            STATE.driver, min_year=min_year, max_year=max_year,
            entity_type=etype, limit=limit,
        ))
    if min_value is not None or max_value is not None or property_name or unit:
        results.extend(await entities_by_numeric_value(
            STATE.driver, property_name=property_name, min_value=min_value,
            max_value=max_value, unit=unit, entity_type=etype, limit=limit,
        ))
    return {"results": results[:limit], "count": len(results[:limit])}


@app.post("/route", response_model=RouteOnlyResponse, dependencies=[Depends(require_api_key)])
async def route_only(req: RouteOnlyRequest) -> RouteOnlyResponse:
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    STATE.request_count += 1
    if STATE.dispatcher is None:
        raise HTTPException(status_code=503, detail="Dispatcher not initialized")
    try:
        from routing import build_query_router
        router = build_query_router(driver=STATE.driver)
        decision = await router.route(req.query)
    except Exception as exc:
        STATE.error_count += 1
        log.exception("Routing failed for query=%r", req.query)
        raise HTTPException(status_code=500, detail=f"routing_error: {exc}") from exc

    analysis = decision.query_analysis
    markers: dict[str, bool] = {}
    if analysis is not None:
        markers = {
            "numeric": analysis.has_numeric_constraint,
            "geography": analysis.has_geo_marker,
            "temporal": analysis.has_temporal_marker,
            "definitional": analysis.is_definitional,
            "causal": analysis.is_causal,
            "comparison": analysis.is_comparison,
        }
    return RouteOnlyResponse(
        query=req.query,
        route=str(decision.route.value),
        confidence=round(decision.confidence, 4),
        coverage_score=round(decision.coverage_score, 4),
        ambiguity_score=round(decision.ambiguity_score, 4),
        reasons=list(decision.reasons),
        markers=markers,
        request_id=request_id,
    )


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    STATE.request_count += 1
    if STATE.dispatcher is None:
        raise HTTPException(status_code=503, detail="Dispatcher not initialized")

    try:
        result: DispatchResult = await STATE.dispatcher.dispatch(
            req.query, synthesize=req.synthesize,
        )
    except Exception as exc:
        STATE.error_count += 1
        log.exception("Dispatch failed for query=%r", req.query)
        raise HTTPException(status_code=500, detail=f"dispatch_error: {exc}") from exc

    answer = result.synthesis.answer if result.synthesis is not None else ""
    used_llm = bool(result.synthesis and result.synthesis.used_llm)
    if result.synthesis is not None:
        STATE.synthesis_calls += 1

    rag_docs: list[dict[str, Any]] = []
    if result.rag_result is not None:
        for doc in result.rag_result.documents[: req.top_k]:
            rag_docs.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "snippet": doc.snippet,
                "score": doc.score,
                "source": doc.source,
                "matched_entities": list(doc.matched_entities),
            })

    return QueryResponse(
        query=req.query,
        route=str(result.decision.route.value),
        confidence=round(result.decision.confidence, 4),
        coverage_score=round(result.decision.coverage_score, 4),
        ambiguity_score=round(result.decision.ambiguity_score, 4),
        reasons=list(result.decision.reasons),
        answer=answer,
        used_llm=used_llm,
        graph_text=result.graph_text,
        rag_documents=rag_docs,
        request_id=request_id,
    )


@app.post(
    "/deep-research",
    response_model=DeepResearchResponse,
    dependencies=[Depends(require_api_key)],
)
async def deep_research_endpoint(req: DeepResearchRequest) -> DeepResearchResponse:
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    STATE.request_count += 1
    if STATE.deep_research_agent is None:
        raise HTTPException(status_code=503, detail="Deep research agent not initialized")

    try:
        result = await STATE.deep_research_agent.run(
            req.query,
            max_iterations=req.max_iterations,
        )
    except Exception as exc:
        STATE.error_count += 1
        log.exception("Deep research failed for query=%r", req.query)
        raise HTTPException(status_code=500, detail=f"deep_research_error: {exc}") from exc

    if result.used_llm:
        STATE.synthesis_calls += 1

    iterations: list[DeepResearchIterationResponse] = []
    for iteration in result.iterations:
        rag_docs: list[dict[str, Any]] = []
        for doc in iteration.rag_documents:
            rag_docs.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "snippet": doc.snippet,
                "score": doc.score,
                "source": doc.source,
                "matched_entities": list(doc.matched_entities),
            })
        iterations.append(
            DeepResearchIterationResponse(
                index=iteration.index,
                query=iteration.query,
                route=iteration.route,
                confidence=iteration.confidence,
                coverage_score=iteration.coverage_score,
                graph_text=iteration.graph_text,
                rag_documents=rag_docs,
                notes=list(iteration.notes),
            )
        )

    return DeepResearchResponse(
        query=result.query,
        answer=result.answer,
        used_llm=result.used_llm,
        model_uri=result.model_uri,
        iterations=iterations,
        follow_up_queries=result.follow_up_queries,
        sources=result.sources,
        notes=list(result.notes),
        request_id=request_id,
    )


@app.get("/metrics")
async def metrics() -> str:
    """Prometheus-style plain text metrics (no auth)."""
    lines = [
        "# HELP nk_uptime_seconds Process uptime in seconds",
        "# TYPE nk_uptime_seconds counter",
        f"nk_uptime_seconds {round(time.time() - STATE.started_at, 2)}",
        "# HELP nk_requests_total Total number of /query and /route calls",
        "# TYPE nk_requests_total counter",
        f"nk_requests_total {STATE.request_count}",
        "# HELP nk_errors_total Total number of failed dispatches",
        "# TYPE nk_errors_total counter",
        f"nk_errors_total {STATE.error_count}",
    ]
    if STATE.driver is not None:
        try:
            stats = await graph_statistics(STATE.driver)
            for label, count in stats.items():
                lines.append(f"nk_graph_{label} {count}")
        except Exception:  # pragma: no cover
            pass
    return "\n".join(lines) + "\n"


__all__ = ["app"]
