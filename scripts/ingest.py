"""
End-to-end ingestion script.

Walks a directory of parsed Markdown files, chunks each file,
runs Natasha + YandexGPT NER+RE, merges via ``EnsembleMerger``,
and loads the result into Neo4j.

Designed to be runnable in stages so a long ingestion can be
resumed after a crash:

  python -m scripts.ingest                 # default: full pipeline
  python -m scripts.ingest --skip-llm      # use Natasha only (no LLM cost)
  python -m scripts.ingest --skip-neo4j    # write JSONL only, don't load
  python -m scripts.ingest --limit 50      # process only 50 files
  python -m scripts.ingest --input Статьи  # custom input dir

Outputs:
  parsed_chunks/ingest_report.json        — summary
  parsed_chunks/merged.jsonl              — merged entities + relations
  Neo4j                                    — final loaded graph
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Windows console defaults to cp1251; force UTF-8 so Cyrillic and symbols in
# help/log output do not fail with UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env before create_llm_client() reads os.environ. Inline environment
# variables still win over values from .env.
try:  # pragma: no cover - environment-specific
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:  # pragma: no cover - optional dependency/runtime setup
    pass

from chunking.chunker import build_raw_chunks
from chunking.config import default_config as default_chunk_config
from chunking.natasha_pipeline import get_pipeline

from ensemble import EnsembleMerger, EnsembleResult
from llm_pipeline_fewshot.llm_parser import (
    ChunkExtractor,
    ChunkInput,
    YandexGPTError,
    create_llm_client,
)
from llm_pipeline_fewshot.models import (
    EnrichedEntity,
    EnrichedRelation,
    EntityType,
    ChunkProvenance as LLMChunkProvenance,
)
from neo4j_integration.neo4j_config import Neo4jConfig
from neo4j_integration.neo4j_loader import Neo4jLoader
from synonym_normalization.canonicalizer import canonicalize_text
from synonym_normalization.normalize_pipeline import (
    normalize_entities,
    normalize_relations,
)
from synonym_normalization.synonym_dictionary import SynonymDictionary

log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "Статьи"
DEFAULT_OUTPUT = REPO_ROOT / "parsed_chunks"


@dataclass
class IngestReport:
    started_at: float
    finished_at: float = 0.0
    files_total: int = 0
    files_processed: int = 0
    files_empty: int = 0
    files_errored: int = 0
    chunks_total: int = 0
    entities_total: int = 0
    relations_total: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": round(self.finished_at - self.started_at, 2),
            "files_total": self.files_total,
            "files_processed": self.files_processed,
            "files_empty": self.files_empty,
            "files_errored": self.files_errored,
            "chunks_total": self.chunks_total,
            "entities_total": self.entities_total,
            "relations_total": self.relations_total,
            "by_source": dict(self.by_source),
            "skipped": list(self.skipped),
        }


# ---------------------------------------------------------------------------
# Step 1: walk input dir → chunks
# ---------------------------------------------------------------------------


def discover_markdown_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".md":
        return [root]
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def chunk_file(path: Path, pipeline: Any) -> list[ChunkInput]:
    """Chunk a single Markdown file into ``ChunkInput`` objects."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1251", errors="replace")

    if not text.strip():
        return []

    cfg = default_chunk_config()
    raws = build_raw_chunks(text, pipeline, cfg)
    rel = path.name
    out: list[ChunkInput] = []
    for idx, rc in enumerate(raws):
        out.append(ChunkInput(
            chunk_id=f"{rel}#{idx:04d}",
            index=idx,
            provenance=LLMChunkProvenance(
                source_document=rel,
                char_start=rc.char_start,
                char_end=rc.char_end,
                heading_path=list(rc.heading_path or []),
            ),
            text=rc.text,
            overlap_prefix_chars=rc.overlap_prefix_chars,
            oversize=rc.oversize,
        ))
    return out


# ---------------------------------------------------------------------------
# Step 2: Natasha entities (free) + LLM entities (paid) → ensemble
# ---------------------------------------------------------------------------


def natasha_entities_for_chunk(
    chunk: ChunkInput,
    pipeline: Any,
) -> list[EnrichedEntity]:
    """Reuse the chunking Natasha pipeline to produce ``EnrichedEntity`` objects."""
    try:
        annotation = pipeline.annotate(chunk.text)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Natasha annotation failed on %s: %s", chunk.chunk_id, exc)
        return []
    entities: list[EnrichedEntity] = []
    for idx, primary in enumerate(annotation.primary_entities or []):
        text = (primary.text or "").strip()
        if not text:
            continue
        try:
            entities.append(EnrichedEntity(
                entity=text,
                type=_natasha_to_entity_type(str(primary.type)),
                chunk_id=chunk.chunk_id,
                source_document=chunk.provenance.source_document,
                page=chunk.provenance.page,
                quote=text,
                confidence=0.6,
                local_id=f"n_{chunk.chunk_id}_{idx}",
                mentions=[text],
                char_start=int(primary.start) + chunk.provenance.char_start,
                char_end=int(primary.stop) + chunk.provenance.char_start,
                extractor="natasha",
            ))
        except Exception as exc:  # pragma: no cover
            log.debug("Skipping bad Natasha entity: %s", exc)
    return entities


_NATASHA_TYPE_MAP = {
    "PER": EntityType.EXPERT,
    "LOC": EntityType.FACILITY,
    "ORG": EntityType.ORGANIZATION,
}


def _natasha_to_entity_type(natasha_type: str) -> EntityType:
    return _NATASHA_TYPE_MAP.get(natasha_type, EntityType.SUBSTANCE)


def llm_extract_chunk(
    chunk: ChunkInput,
    extractor: ChunkExtractor | None,
) -> tuple[list[EnrichedEntity], list[EnrichedRelation]]:
    """Run the LLM extractor on a chunk. Returns ([], []) on failure."""
    if extractor is None:
        return [], []
    try:
        result = extractor.extract_chunk(chunk)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("LLM extraction failed on %s: %s: %s", chunk.chunk_id, type(exc).__name__, exc)
        return [], []
    if result.status != "ok":
        return [], []
    return list(result.entities), list(result.relations)


def ensemble_chunk(
    chunk: ChunkInput,
    natasha_ents: list[EnrichedEntity],
    llm_ents: list[EnrichedEntity],
    llm_rels: list[EnrichedRelation],
    merger: EnsembleMerger,
    syn_dict: SynonymDictionary,
) -> tuple[list[EnrichedEntity], list[EnrichedRelation]]:
    """Run normalization + ensemble merge on a single chunk's outputs."""
    if natasha_ents or llm_ents:
        normalize_entities(natasha_ents + llm_ents, syn_dict)
    if llm_rels:
        normalize_relations(llm_rels, syn_dict)

    merged = merger.merge(natasha_ents, llm_ents, [], llm_rels)
    return list(merged.entities), list(merged.relations)


# ---------------------------------------------------------------------------
# Step 3: load into Neo4j
# ---------------------------------------------------------------------------


async def load_to_neo4j(
    entities: Iterable[EnrichedEntity],
    relations: Iterable[EnrichedRelation],
    *,
    uri: str,
    user: str,
    password: str,
    batch_size: int = 500,
) -> tuple[int, int]:
    """Push all entities and relations into Neo4j in batches."""
    entity_list = list(entities)
    relation_list = list(relations)

    config = Neo4jConfig(uri=uri, user=user, password=password, batch_size=batch_size)
    async with Neo4jLoader(config) as loader:
        await loader.setup_constraints()
        await loader.load_entities(entity_list)
        await loader.load_relations(relation_list)

    return len(entity_list), len(relation_list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def ingest(
    *,
    input_dir: Path,
    output_dir: Path,
    skip_llm: bool,
    skip_neo4j: bool,
    limit: int | None,
    neo4j_uri: str | None,
    neo4j_user: str | None,
    neo4j_password: str | None,
    progress_every: int,
) -> IngestReport:
    report = IngestReport(started_at=time.time())

    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        report.finished_at = time.time()
        return report

    files = discover_markdown_files(input_dir)
    if limit is not None:
        files = files[:limit]
    report.files_total = len(files)
    if not files:
        log.warning("No .md files under %s", input_dir)
        report.finished_at = time.time()
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ingest_report.json"
    merged_path = output_dir / "merged.jsonl"

    pipeline = get_pipeline()
    syn_dict = SynonymDictionary()
    merger = EnsembleMerger()

    extractor: ChunkExtractor | None = None
    if not skip_llm:
        try:
            client = create_llm_client()
            extractor = ChunkExtractor(client=client)
        except YandexGPTError as exc:
            log.warning("LLM provider unavailable (%s); falling back to Natasha-only", exc)
            report.skipped = report.skipped + ("llm_unavailable",)
            extractor = None

    all_entities: list[EnrichedEntity] = []
    all_relations: list[EnrichedRelation] = []
    by_source: Counter[str] = Counter()

    with merged_path.open("w", encoding="utf-8") as out_jsonl:
        for index, file in enumerate(files, start=1):
            try:
                chunks = chunk_file(file, pipeline)
            except Exception as exc:
                log.warning("Chunking failed for %s: %s", file.name, exc)
                report.files_errored += 1
                continue

            if not chunks:
                report.files_empty += 1
                continue

            for chunk in chunks:
                natasha_ents = natasha_entities_for_chunk(chunk, pipeline)
                llm_ents, llm_rels = llm_extract_chunk(chunk, extractor)
                entities, relations = ensemble_chunk(
                    chunk, natasha_ents, llm_ents, llm_rels, merger, syn_dict,
                )
                for entity in entities:
                    all_entities.append(entity)
                    by_source[str(entity.extractor)] += 1
                for relation in relations:
                    all_relations.append(relation)
                    by_source[str(relation.extractor)] += 1

                # Stream to JSONL for resumability + debugging.
                rec = {
                    "chunk_id": chunk.chunk_id,
                    "source_document": chunk.provenance.source_document,
                    "char_start": chunk.provenance.char_start,
                    "char_end": chunk.provenance.char_end,
                    "n_entities": len(entities),
                    "n_relations": len(relations),
                    "entities": [entity.model_dump() for entity in entities],
                    "relations": [relation.model_dump() for relation in relations],
                }
                out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

            report.files_processed += 1
            report.chunks_total += len(chunks)
            if index % progress_every == 0 or index == len(files):
                # Use the live list length (single source of truth) rather than
                # a cumulative running counter; the previous ``+= sum(1 for _
                # in all_entities)`` here produced a non-monotonic growing
                # value because ``all_entities`` itself grows each iteration.
                log.info(
                    "[ingest] %d/%d files | %d chunks | %d entities | %d relations",
                    index, len(files), report.chunks_total,
                    len(all_entities), len(all_relations),
                )

    report.entities_total = len(all_entities)
    report.relations_total = len(all_relations)
    report.by_source = dict(by_source)

    if not skip_neo4j and (neo4j_uri or all_entities):
        try:
            loaded_e, loaded_r = asyncio.run(load_to_neo4j(
                all_entities, all_relations,
                uri=neo4j_uri or Neo4jConfig().uri,
                user=neo4j_user or Neo4jConfig().user,
                password=neo4j_password or Neo4jConfig().password,
            ))
            log.info("[ingest] loaded %d entities and %d relations into Neo4j", loaded_e, loaded_r)
        except Exception as exc:
            log.warning("Neo4j load failed: %s: %s", type(exc).__name__, exc)
            report.skipped = report.skipped + (f"neo4j_load_failed:{type(exc).__name__}",)

    report.finished_at = time.time()
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[ingest] done in %.1fs: %s", report.finished_at - report.started_at, report_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.ingest",
        description="End-to-end ingestion: parse → chunk → Natasha + LLM → ensemble → Neo4j",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Root directory of .md files (default: ./Статьи)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for merged.jsonl + report (default: ./parsed_chunks)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N files",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Use Natasha only (saves LLM cost, lower recall)",
    )
    parser.add_argument(
        "--skip-neo4j", action="store_true",
        help="Don't load to Neo4j; only write merged.jsonl",
    )
    parser.add_argument(
        "--neo4j-uri", default=None,
        help="Neo4j URI (default: env NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j-user", default=None,
        help="Neo4j user (default: env NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--neo4j-password", default=None,
        help="Neo4j password (default: env NEO4J_PASSWORD)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=10,
        help="Log progress every N files",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report = ingest(
        input_dir=args.input,
        output_dir=args.output,
        skip_llm=args.skip_llm,
        skip_neo4j=args.skip_neo4j,
        limit=args.limit,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        progress_every=args.progress_every,
    )

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "chunk_file",
    "discover_markdown_files",
    "ensemble_chunk",
    "ingest",
    "load_to_neo4j",
    "natasha_entities_for_chunk",
]


if __name__ == "__main__":
    raise SystemExit(main())
