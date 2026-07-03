"""CLI entry-point for the query dispatcher.

Usage::

    # Print the dispatch result with synthesized answer as Markdown
    python -m agent.cli "Какие методы обессоливания воды при сульфатах ≤300 мг/л?"

    # Print the routing decision only (no graph/RAG execution, no synthesis)
    python -m agent.cli --decision-only "никель электроэкстракция"

    # Output JSON (for programmatic consumers)
    python -m agent.cli --json "покажи эксперименты по флотации"

    # Disable LLM synthesis (render graph context directly)
    python -m agent.cli --no-synthesis "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Sequence

from graph_reasoning.neo4j_subgraph import Neo4jSubgraphExtractor
from llm_pipeline_fewshot.llm_parser import YandexGPTClient
from neo4j_integration.neo4j_config import Neo4jConfig
from neo4j_integration.neo4j_loader import Neo4jLoader
from routing import build_query_router

from .dispatcher import Dispatcher
from .rag_client import RAGClient, StubRAGClient
from .synthesizer import AnswerSynthesizer


def _build_dispatcher(
    rag_client: RAGClient | None = None,
    driver: Any | None = None,
    synthesizer: AnswerSynthesizer | None = None,
    no_neo4j: bool = False,
    no_synthesis: bool = False,
) -> Dispatcher:
    """Construct a ``Dispatcher`` with sensible defaults.

    When ``no_neo4j`` is True (the default for offline / demo runs)
    the dispatcher is built without a graph backend, so every query
    routes to ``RAG_ONLY`` and the dispatcher only calls the RAG
    client. This is useful for smoke tests and for showing the
    router's behaviour before the corpus is loaded.

    When ``no_synthesis`` is True the dispatcher is built without
    an LLM synthesizer; the markdown output will contain only the
    structured graph/RAG context, not a natural-language answer.
    """
    router = build_query_router(driver=driver)
    extractor = Neo4jSubgraphExtractor(driver) if driver is not None else None
    synth = None
    if not no_synthesis and synthesizer is not None:
        synth = synthesizer
    elif not no_synthesis:
        try:
            client = YandexGPTClient()
            synth = AnswerSynthesizer(client=client)
        except Exception:
            synth = None  # fall back to context-only render
    return Dispatcher(
        router=router,
        graph_extractor=extractor,
        rag_client=rag_client or StubRAGClient(),
        synthesizer=synth,
    )


def _format_decision_json(dispatcher_result: Any) -> str:
    decision = dispatcher_result.decision
    payload = decision.to_dict()
    payload["query"] = dispatcher_result.query
    payload["notes"] = list(dispatcher_result.notes)
    if dispatcher_result.graph_text is not None:
        payload["graph_text"] = dispatcher_result.graph_text
    if dispatcher_result.rag_result is not None:
        rag = dispatcher_result.rag_result
        payload["rag"] = {
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "score": doc.score,
                    "source": doc.source,
                    "matched_entities": list(doc.matched_entities),
                    "snippet": doc.snippet,
                }
                for doc in rag.documents
            ],
            "notes": list(rag.notes),
        }
    if dispatcher_result.synthesis is not None:
        payload["synthesis"] = {
            "answer": dispatcher_result.synthesis.answer,
            "used_llm": dispatcher_result.synthesis.used_llm,
            "model_uri": dispatcher_result.synthesis.model_uri,
            "input_tokens": dispatcher_result.synthesis.input_tokens,
            "output_tokens": dispatcher_result.synthesis.output_tokens,
            "error": dispatcher_result.synthesis.error,
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def _run_with_neo4j(query: str, args: argparse.Namespace) -> Any:
    config = Neo4jConfig()
    async with Neo4jLoader(config) as loader:
        dispatcher = _build_dispatcher(
            rag_client=args.rag_client,
            driver=loader.driver,
            no_synthesis=args.no_synthesis,
        )
        return await dispatcher.dispatch(query, synthesize=not args.no_synthesis)


async def _run_offline(query: str, args: argparse.Namespace) -> Any:
    dispatcher = _build_dispatcher(
        rag_client=args.rag_client,
        no_neo4j=True,
        no_synthesis=args.no_synthesis,
    )
    return await dispatcher.dispatch(query, synthesize=not args.no_synthesis)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent.cli",
        description="Query dispatcher CLI for the Nornikel Knowledge Graph.",
    )
    parser.add_argument("query", help="User query in natural language.")
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Connect to Neo4j and run the full dispatcher (requires env vars).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the dispatch result as JSON instead of Markdown.",
    )
    parser.add_argument(
        "--decision-only",
        action="store_true",
        help="Print the routing decision only (no graph/RAG execution).",
    )
    parser.add_argument(
        "--no-synthesis",
        action="store_true",
        help="Skip LLM answer synthesis; render the raw context only.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.decision_only:
        from routing import build_query_router
        router = build_query_router()
        decision = asyncio.run(router.route(args.query))
        sys.stdout.write(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        return 0

    runner = _run_with_neo4j if args.neo4j else _run_offline
    try:
        result = asyncio.run(runner(args.query, args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        sys.stderr.write(f"error: {type(exc).__name__}: {exc}\n")
        return 1

    if args.json:
        sys.stdout.write(_format_decision_json(result))
    else:
        sys.stdout.write(result.to_markdown())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
