from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .chunker import build_raw_chunks
from .config import PARSED_TEXTS_DIR, WORKERS, ChunkConfig, default_config
from .orchestrator import run

def _cfg_from_args(args) -> ChunkConfig:
    base = default_config()
    return ChunkConfig(
        target_chunk_chars=args.target or base.target_chunk_chars,
        max_chunk_chars=args.max or base.max_chunk_chars,
        min_chunk_chars=base.min_chunk_chars,
        overlap_sentences=args.overlap if args.overlap is not None else base.overlap_sentences,
        overlap_max_chars=base.overlap_max_chars,
        section_break_level=base.section_break_level,
    )

def _sample(path: Path, cfg: ChunkConfig) -> None:
    from .natasha_pipeline import get_pipeline
    text = path.read_text(encoding="utf-8")
    raws = build_raw_chunks(text, get_pipeline(), cfg)
    print(f"{path}: {len(raws)} chunk(s)\n")
    for i, rc in enumerate(raws):
        preview = rc.text[:160].replace("\n", " ")
        print(f"--- chunk {i:03d} | [{rc.char_start}:{rc.char_end}] "
              f"({rc.char_end - rc.char_start} chars, overlap {rc.overlap_prefix_chars}"
              f"{', OVERSIZE' if rc.oversize else ''}) | {' > '.join(rc.heading_path)}")
        print(f"    {preview}...")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PARSED_TEXTS_DIR), help="root of parsed .md files")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N documents")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--target", type=int, default=None, help="target chunk size in chars")
    ap.add_argument("--max", type=int, default=None, help="hard max chunk size in chars")
    ap.add_argument("--overlap", type=int, default=None, help="overlap sentences")
    ap.add_argument("--sample", default=None, help="chunk ONE file and print summary (no write)")
    args = ap.parse_args()

    cfg = _cfg_from_args(args)
    if args.sample:
        _sample(Path(args.sample), cfg)
        return
    run(cfg=cfg, only=args.limit, root=Path(args.input), workers=args.workers)

if __name__ == "__main__":
    main()
