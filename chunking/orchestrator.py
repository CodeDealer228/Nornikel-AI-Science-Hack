from __future__ import annotations

import json
import logging
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from .chunker import build_raw_chunks, strip_front_matter
from .config import (
    CHUNK_REPORT_PATH,
    CHUNKS_PATH,
    PARSED_TEXTS_DIR,
    WORKERS,
    ChunkConfig,
    default_config,
)
from .models import Chunk, ChunkProvenance

log = logging.getLogger(__name__)

def discover_files(root: Path = PARSED_TEXTS_DIR) -> List[Path]:
    return sorted(p for p in root.rglob("*.md") if p.is_file())

def process_file(path_str: str, cfg: ChunkConfig, root_str: str) -> Tuple[dict, List[dict]]:
    from .natasha_pipeline import get_pipeline

    path = Path(path_str)
    root = Path(root_str)
    rel = path.relative_to(root).as_posix()
    entry = {"file": rel, "status": "ok", "n_chunks": 0, "oversize": 0, "ner_available": None}
    try:
        text = path.read_text(encoding="utf-8")
        _, doc_meta = strip_front_matter(text)

        pipeline = get_pipeline()
        raws = build_raw_chunks(text, pipeline, cfg)

        chunk_dicts: List[dict] = []
        ner_flag = None
        for idx, rc in enumerate(raws):
            ann = pipeline.annotate(rc.text)
            ner_flag = ann.ner_available
            chunk = Chunk(
                chunk_id=f"{rel}#{idx:04d}",
                index=idx,
                provenance=ChunkProvenance(
                    source_document=rel,
                    char_start=rc.char_start,
                    char_end=rc.char_end,
                    heading_path=rc.heading_path,
                ),
                text=rc.text,
                overlap_prefix_chars=rc.overlap_prefix_chars,
                oversize=rc.oversize,
                natasha=ann,
                doc_metadata=doc_meta,
            )
            chunk_dicts.append(chunk.model_dump())

        entry["n_chunks"] = len(chunk_dicts)
        entry["oversize"] = sum(1 for c in chunk_dicts if c["oversize"])
        entry["ner_available"] = ner_flag
        if not chunk_dicts:
            entry["status"] = "empty"
        return entry, chunk_dicts
    except Exception as e:
        entry["status"] = "error"
        entry["reason"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc(limit=5)
        return entry, []

def run(cfg: ChunkConfig | None = None, only: int | None = None,
        root: Path = PARSED_TEXTS_DIR, workers: int = WORKERS) -> dict:
    cfg = cfg or default_config()
    files = discover_files(root)
    if only is not None:
        files = files[:only]
    if not files:
        log.warning("no .md files under %s — run `python -m parsing.run` first", root)
        return {"total": 0, "by_status": {}, "total_chunks": 0}

    print(f"[chunk] {len(files)} document(s) under {root} | workers={workers} | "
          f"target={cfg.target_chunk_chars} overlap={cfg.overlap_sentences}s")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries: List[dict] = []
    total_chunks = 0
    done = 0

    with open(CHUNKS_PATH, "w", encoding="utf-8") as out, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_file, str(f), cfg, str(root)): f for f in files
        }
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                entry, chunk_dicts = fut.result()
            except Exception as e:
                entry = {"file": Path(f).relative_to(root).as_posix(), "status": "error",
                         "reason": f"worker crash: {type(e).__name__}: {e}", "n_chunks": 0}
                chunk_dicts = []
            for cd in chunk_dicts:
                out.write(json.dumps(cd, ensure_ascii=False) + "\n")
            total_chunks += len(chunk_dicts)
            entries.append(entry)
            done += 1
            if done % 25 == 0 or done == len(files):
                print(f"[chunk] {done}/{len(files)} (last: {entry['status']} "
                      f"{entry['file']} -> {entry.get('n_chunks', 0)} chunks)", flush=True)

    report = _write_report(entries, total_chunks, cfg)
    print(f"\n[chunk] done: {report['by_status']} | {total_chunks} chunks -> {CHUNKS_PATH}")
    if report.get("ner_unavailable_files"):
        print(f"[chunk] NOTE: NER models unavailable for {report['ner_unavailable_files']} file(s) "
              f"(segmentation-only) — check Natasha model download/network")
    return report

def _write_report(entries: List[dict], total_chunks: int, cfg: ChunkConfig) -> dict:
    by_status = Counter(e["status"] for e in entries)
    oversize = sum(e.get("oversize", 0) for e in entries)
    ner_unavail = sum(1 for e in entries if e.get("ner_available") is False)
    report = {
        "total": len(entries),
        "by_status": dict(by_status),
        "total_chunks": total_chunks,
        "oversize_chunks": oversize,
        "ner_unavailable_files": ner_unavail,
        "config": cfg.__dict__,
        "entries": entries,
    }
    CHUNK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
