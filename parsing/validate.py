"""
Post-parse validation — runs against every successfully parsed document, not
a sample. Every check is non-fatal: it appends a flag to the document's
report entry rather than raising, per the existing "fail loud per item, keep
the batch going" convention (CLAUDE.md Core Principles / a.py's error
handling). The signature gate (signatures.py) runs separately, before
parsing even starts — these checks validate a parse's *output*.
"""
from pathlib import Path
from typing import Dict, List

from .common import ParsedDocument
from .config import ENCODING_SUSPECT_RATIO, LOW_YIELD_CHARS_PER_UNIT


def validate(parsed: ParsedDocument, image_dir: Path) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []

    referenced = set(parsed.image_ids())
    on_disk = set()
    if image_dir.is_dir():
        on_disk = {p.stem.replace("IMAGE_", "") for p in image_dir.iterdir() if p.stem.startswith("IMAGE_")}
    missing_files = referenced - on_disk
    orphan_files = on_disk - referenced
    if missing_files:
        flags.append({
            "check": "IMAGE_MISMATCH",
            "detail": f"{len(missing_files)} placeholder(s) with no file on disk: {sorted(missing_files)[:10]}",
        })
    if orphan_files:
        flags.append({
            "check": "IMAGE_MISMATCH",
            "detail": f"{len(orphan_files)} file(s) on disk with no placeholder: {sorted(orphan_files)[:10]}",
        })

    if parsed.doc_type != "image":
        # image passthrough deliberately extracts zero text (no OCR in this
        # pass) — flagging that every time would just restate the design.
        text_len = parsed.text_length()
        units = max(parsed.units_total, 1)
        avg = text_len / units
        if avg < LOW_YIELD_CHARS_PER_UNIT:
            flags.append({
                "check": "LOW_YIELD",
                "detail": f"avg {avg:.1f} chars/unit over {units} unit(s), total {text_len} chars — likely NO_TEXT_LAYER",
            })

    full_text = "".join(str(b.content) for b in parsed.blocks if b.kind in ("text", "heading") and b.content)
    if full_text:
        suspect = sum(1 for ch in full_text if ch == "�" or (ord(ch) < 32 and ch not in "\n\t\r"))
        ratio = suspect / len(full_text)
        if ratio > ENCODING_SUSPECT_RATIO:
            flags.append({"check": "ENCODING_SUSPECT", "detail": f"{ratio:.1%} suspect/replacement characters"})

    dropped = parsed.metadata.get("images_dropped_below_floor")
    if dropped:
        flags.append({"check": "IMAGES_DROPPED", "detail": f"{dropped} sub-floor image(s) dropped (informational)"})

    if parsed.metadata.get("images_skipped") == "XLS_IMAGES_SKIPPED":
        flags.append({"check": "XLS_IMAGES_SKIPPED", "detail": "legacy .xls: image extraction not attempted (known gap)"})

    return flags
