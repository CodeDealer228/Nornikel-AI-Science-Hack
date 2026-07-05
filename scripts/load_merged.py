"""Load already-produced parsed_chunks/*/merged.jsonl batches into Neo4j.

Unlike ``scripts.ingest``, this does not re-parse or call an LLM — it just
loads the entities/relations already sitting in ``merged.jsonl`` files
(produced by a prior ``scripts.ingest`` run) into a running Neo4j instance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation
from neo4j_integration.neo4j_config import Neo4jConfig
from neo4j_integration.neo4j_loader import Neo4jLoader

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = [
    REPO / "parsed_chunks" / "articles" / "merged.jsonl",
    REPO / "parsed_chunks" / "reports" / "merged.jsonl",
]


def load_records(paths: Sequence[Path]) -> tuple[list[EnrichedEntity], list[EnrichedRelation]]:
    entities: list[EnrichedEntity] = []
    relations: list[EnrichedRelation] = []
    for path in paths:
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for e in record.get("entities") or []:
                    entities.append(EnrichedEntity(**e))
                for r in record.get("relations") or []:
                    relations.append(EnrichedRelation(**r))
        print(f"loaded from {path}: running totals -> {len(entities)} entities, {len(relations)} relations")
    return entities, relations


async def load_to_neo4j(config: Neo4jConfig, entities: list[EnrichedEntity], relations: list[EnrichedRelation]) -> None:
    async with Neo4jLoader(config) as loader:
        await loader.setup_constraints()
        await loader.load_entities(entities)
        await loader.load_relations(relations)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load merged.jsonl batches into Neo4j")
    parser.add_argument("--input", type=Path, nargs="*", default=None, help="merged.jsonl paths (default: articles + reports)")
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    args = parser.parse_args(argv)

    paths = args.input or DEFAULT_INPUTS
    entities, relations = load_records(paths)

    config = Neo4jConfig()
    if args.neo4j_uri:
        config.uri = args.neo4j_uri
    if args.neo4j_user:
        config.user = args.neo4j_user
    if args.neo4j_password:
        config.password = args.neo4j_password

    asyncio.run(load_to_neo4j(config, entities, relations))
    print(f"done: loaded {len(entities)} entities and {len(relations)} relations into {config.uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
