from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PARSED_TEXTS_DIR = Path("parsed_data") / "texts"
OUTPUT_DIR = Path("parsed_data")
CHUNKS_PATH = OUTPUT_DIR / "chunks.jsonl"
CHUNK_REPORT_PATH = OUTPUT_DIR / "chunk_report.json"

WORKERS = 4

ENABLE_NER = True

STORE_LEMMAS = True
MAX_LEMMAS_STORED = 0

@dataclass(frozen=True)
class ChunkConfig:
    target_chunk_chars: int = 2500
    max_chunk_chars: int = 4000
    min_chunk_chars: int = 400
    overlap_sentences: int = 2
    overlap_max_chars: int = 400
    section_break_level: int = 2

def default_config() -> ChunkConfig:
    return ChunkConfig()
