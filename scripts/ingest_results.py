"""Convert precomputed NER/RE results to graph-ready JSONL.

This script starts from ``ner_re_extraction/result``. It does not call an LLM
and does not rerun chunking. Input records are expected to look like:

    {"chunk_id": "...", "doc_id": "...", "parsed": {"entities": [], "relations": []}}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:  # pragma: no cover - optional runtime convenience
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    pass

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation
from neo4j_integration.neo4j_config import Neo4jConfig
from neo4j_integration.neo4j_loader import Neo4jLoader
from synonym_normalization.normalize_pipeline import normalize_entities, normalize_relations
from synonym_normalization.synonym_dictionary import SynonymDictionary


DEFAULT_INPUT = Path("ner_re_extraction") / "result"
DEFAULT_OUTPUT = Path("parsed_chunks") / "ner_re_results"


@dataclass
class Report:
    input_paths: list[str] = field(default_factory=list)
    output_dir: str = ""
    records_total: int = 0
    records_loaded: int = 0
    records_skipped: int = 0
    invalid_json: int = 0
    invalid_entities: int = 0
    invalid_relations: int = 0
    entities_total: int = 0
    relations_total: int = 0
    source_documents: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    neo4j_loaded: bool = False
    neo4j_error: str | None = None
    duration_sec: float = 0.0


def discover_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"input path does not exist: {path}")
    return sorted(p for p in path.rglob("*.jsonl") if p.is_file())


def model_dump(obj: Any) -> dict[str, Any]:
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def convert_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, int, int]:
    parsed = record.get("parsed") or {}
    if not isinstance(parsed, dict):
        return None, 0, 0

    status = str(record.get("status") or "OK").upper()
    if status not in {"OK", "SUCCESS", ""}:
        return None, 0, 0

    chunk_id = str(record.get("chunk_id") or "").strip()
    doc_id = str(record.get("doc_id") or record.get("source_document") or "").strip()
    if not chunk_id or not doc_id:
        return None, 0, 0

    validation = record.get("validation") or {}
    needs_review = bool(validation.get("flags") or [])
    char_start = safe_int(record.get("char_start"), 0)
    char_end = safe_int(record.get("char_end"), char_start)
    page = record.get("page") if isinstance(record.get("page"), int) else None

    bad_entities = 0
    entities: list[EnrichedEntity] = []
    by_local_id: dict[str, EnrichedEntity] = {}
    for raw in parsed.get("entities") or []:
        if not isinstance(raw, dict):
            bad_entities += 1
            continue
        local_id = str(raw.get("local_id") or "").strip()
        name = str(raw.get("canonical_name") or raw.get("entity") or "").strip()
        mentions = [str(m).strip() for m in (raw.get("mentions") or []) if str(m).strip()]
        if not local_id or not name:
            bad_entities += 1
            continue
        try:
            entity = EnrichedEntity(
                entity=name,
                type=str(raw.get("type") or "").strip(),
                chunk_id=chunk_id,
                source_document=doc_id,
                page=page,
                quote=mentions[0] if mentions else name,
                confidence=0.85,
                local_id=local_id,
                mentions=mentions,
                attributes=raw.get("attributes") or {},
                heading_path=[],
                char_start=char_start,
                char_end=char_end,
                extractor="ner_re_result",
                needs_review=needs_review,
            )
        except Exception:
            bad_entities += 1
            continue
        entities.append(entity)
        by_local_id[local_id] = entity

    bad_relations = 0
    relations: list[EnrichedRelation] = []
    for raw in parsed.get("relations") or []:
        if not isinstance(raw, dict):
            bad_relations += 1
            continue
        subject = by_local_id.get(str(raw.get("subject") or "").strip())
        obj = by_local_id.get(str(raw.get("object") or "").strip())
        if subject is None or obj is None:
            bad_relations += 1
            continue
        try:
            relation = EnrichedRelation(
                source_entity=subject.entity,
                target_entity=obj.entity,
                relation_type=str(raw.get("predicate") or raw.get("relation_type") or "").strip(),
                chunk_id=chunk_id,
                source_document=doc_id,
                page=page,
                quote=str(raw.get("quote") or raw.get("evidence_text") or ""),
                confidence=0.85,
                source_entity_type=subject.type,
                target_entity_type=obj.type,
                source_local_id=subject.local_id,
                target_local_id=obj.local_id,
                heading_path=[],
                char_start=char_start,
                char_end=char_end,
                extractor="ner_re_result",
                needs_review=needs_review,
                note=str(raw.get("note")) if raw.get("note") else None,
            )
        except Exception:
            bad_relations += 1
            continue
        relations.append(relation)

    syn_dict = SynonymDictionary()
    if entities:
        entities = normalize_entities(entities, syn_dict)
    if relations:
        relations = normalize_relations(relations, syn_dict)

    converted = {
        "chunk_id": chunk_id,
        "source_document": doc_id,
        "char_start": char_start,
        "char_end": char_end,
        "entities": [model_dump(entity) for entity in entities],
        "relations": [model_dump(relation) for relation in relations],
        "n_entities": len(entities),
        "n_relations": len(relations),
        "metadata": {
            "source_format": "ner_re_extraction.result",
            "status": record.get("status"),
            "validation": validation,
            "latency_s": record.get("latency_s"),
            "batch_size": record.get("batch_size"),
        },
    }
    return converted, bad_entities, bad_relations


def convert_results(input_path: Path, output_dir: Path, limit: int | None = None) -> tuple[Report, list[EnrichedEntity], list[EnrichedRelation]]:
    started = time.time()
    inputs = discover_inputs(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = Report(input_paths=[str(p) for p in inputs], output_dir=str(output_dir))
    source_docs: set[str] = set()
    by_source: Counter[str] = Counter()
    all_entities: list[EnrichedEntity] = []
    all_relations: list[EnrichedRelation] = []

    with (output_dir / "merged.jsonl").open("w", encoding="utf-8") as out:
        for path in inputs:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if limit is not None and report.records_total >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    report.records_total += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        report.invalid_json += 1
                        report.records_skipped += 1
                        continue
                    converted, bad_entities, bad_relations = convert_record(record)
                    report.invalid_entities += bad_entities
                    report.invalid_relations += bad_relations
                    if converted is None:
                        report.records_skipped += 1
                        continue
                    out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    report.records_loaded += 1
                    report.entities_total += converted["n_entities"]
                    report.relations_total += converted["n_relations"]
                    source = converted["source_document"]
                    source_docs.add(source)
                    by_source[source] += converted["n_entities"]
                    all_entities.extend(EnrichedEntity(**item) for item in converted["entities"])
                    all_relations.extend(EnrichedRelation(**item) for item in converted["relations"])
            if limit is not None and report.records_total >= limit:
                break

    report.source_documents = len(source_docs)
    report.by_source = dict(by_source.most_common())
    report.duration_sec = round(time.time() - started, 3)
    return report, all_entities, all_relations


async def load_to_neo4j(
    config: Neo4jConfig,
    entities: Sequence[EnrichedEntity],
    relations: Sequence[EnrichedRelation],
    *,
    relations_only: bool = False,
) -> None:
    async with Neo4jLoader(config) as loader:
        await loader.setup_constraints()
        if not relations_only:
            await loader.load_entities(entities)
        await loader.load_relations(relations)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert ner_re_extraction/result JSONL to graph-ready merged.jsonl")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument(
        "--neo4j-relations-only",
        action="store_true",
        help="Load only relations into Neo4j; use after entities were already loaded",
    )
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    args = parser.parse_args(argv)

    report, entities, relations = convert_results(args.input, args.output, args.limit)
    if not args.skip_neo4j:
        try:
            config = Neo4jConfig()
            if args.neo4j_uri:
                config.uri = args.neo4j_uri
            if args.neo4j_user:
                config.user = args.neo4j_user
            if args.neo4j_password:
                config.password = args.neo4j_password
            asyncio.run(load_to_neo4j(
                config,
                entities,
                relations,
                relations_only=args.neo4j_relations_only,
            ))
            report.neo4j_loaded = True
        except Exception as exc:
            report.neo4j_error = f"{type(exc).__name__}: {exc}"

    (args.output / "ingest_report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 1 if report.neo4j_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
