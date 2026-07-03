"""
Walks input_docs/Источники информации, dispatches each file through the
signature gate + the right format parser, writes parsed_data/texts +
parsed_data/images, validates the result, and accumulates parse_report.json.

Concurrency split by cost profile:
  - .pdf (1421 files, the long pole) -> ProcessPoolExecutor, CPU-bound
  - .docx/.pptx/.xlsx/.xls/.gif -> ThreadPoolExecutor, light/I-O-ish
  - .doc + .docm (21 files) -> serial, shares one Word COM instance
    (doc_legacy.py). .docm looks like plain OOXML but python-docx rejects it
    outright (`ValueError: ... not a Word file, content type is
    'application/vnd.ms-word.document.macroEnabled.main+xml'` — confirmed
    empirically against all 3 real .docm files in this corpus), so it goes
    through the same Word-COM round-trip as legacy .doc rather than
    python-docx directly. COM automation under concurrency is fragile;
    serial is fast enough at this volume.
"""
import json
import shutil
import tempfile
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List

from . import doc_legacy, docx_parser, image_passthrough, pdf_parser, pptx_parser, xls_parser, xlsx_parser
from . import validate as validate_mod
from .common import ImageWriter
from .config import GENERIC_WORKERS, IMAGES_DIR, INPUT_ROOT, PDF_WORKERS, REPORT_PATH, TEXTS_DIR
from .signatures import check_signature

_EXT_PARSERS: Dict[str, Callable] = {
    ".docx": docx_parser.parse,
    ".pptx": pptx_parser.parse,
    ".xlsx": xlsx_parser.parse,
    ".xls": xls_parser.parse,
    ".gif": lambda p, iw: image_passthrough.parse(p, iw),
}


def discover_files() -> List[Path]:
    return sorted(p for p in INPUT_ROOT.rglob("*") if p.is_file())


def _sanitize_parts(p: Path) -> Path:
    """Strip trailing spaces/periods from every path component.

    Windows silently drops a trailing space/period from a path component
    when it's the *last* segment of a path, but not necessarily when the
    same component is followed by more path — so a source name like
    "22 Резерв .docm" produces an images/ folder name "22 Резерв " (trailing
    space intact once with_suffix("") removes the extension) that `mkdir`
    and a later `write_bytes` can each normalize differently, causing a
    FileNotFoundError despite the mkdir "succeeding" (confirmed empirically
    on this exact file during implementation). Applied to every component,
    not just the last, in case an intermediate folder name has the same
    issue independent of any extension-stripping.
    """
    return Path(*(part.rstrip(" .") or "_" for part in p.parts))


def _paths_for(rel: Path):
    rel = _sanitize_parts(rel)
    md_path = TEXTS_DIR / _sanitize_parts(rel.with_suffix(".md"))
    img_dir = IMAGES_DIR / _sanitize_parts(rel.with_suffix(""))
    return md_path, img_dir


def _build_entry(path: Path, parse_fn: Callable) -> dict:
    """Shared per-file pipeline: signature gate -> parse_fn -> write -> validate.
    Never raises — every failure mode becomes a report entry."""
    rel = path.relative_to(INPUT_ROOT)
    md_path, img_dir = _paths_for(rel)
    entry = {"file": str(rel), "ext": path.suffix.lower(), "status": "ok", "flags": []}
    try:
        sig = check_signature(path)
        image_writer = ImageWriter(img_dir)
        if sig.kind == "image":
            parsed = image_passthrough.parse(path, image_writer, detected_kind=sig.detail)
            entry["dispatched_as"] = f"image({sig.detail})"
        elif sig.kind != "match":
            entry["status"] = "skipped"
            entry["reason"] = f"SIGNATURE_MISMATCH: {sig.kind} {sig.detail}"
            return entry
        else:
            parsed = parse_fn(path, image_writer)

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(parsed.to_markdown(), encoding="utf-8")
        entry["flags"] = validate_mod.validate(parsed, img_dir)
        entry["images"] = image_writer.count
        entry["text_chars"] = parsed.text_length()
    except Exception as e:
        entry["status"] = "error"
        entry["reason"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc(limit=5)
    return entry


def process_generic(path: Path) -> dict:
    ext = path.suffix.lower()
    parse_fn = _EXT_PARSERS.get(ext)
    if parse_fn is None:
        rel = path.relative_to(INPUT_ROOT)
        return {"file": str(rel), "ext": ext, "status": "skipped", "reason": f"no generic handler for {ext}", "flags": []}
    return _build_entry(path, parse_fn)


def process_pdf(path: Path) -> dict:
    """Top-level, picklable — runs inside a ProcessPoolExecutor worker."""
    return _build_entry(path, pdf_parser.parse)


def process_doc_batch(paths: List[Path], converter: doc_legacy.WordConverter, staging_dir: Path) -> List[dict]:
    return [
        _build_entry(p, lambda pp, iw: doc_legacy.parse(pp, iw, converter, staging_dir))
        for p in paths
    ]


def _run_pool(executor_cls, worker_fn, files: List[Path], max_workers: int, label: str) -> List[dict]:
    entries = []
    total = len(files)
    with executor_cls(max_workers=max_workers) as pool:
        futures = {pool.submit(worker_fn, f): f for f in files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            f = futures[fut]
            try:
                entry = fut.result()
            except Exception as e:
                entry = {"file": str(f.relative_to(INPUT_ROOT)), "ext": f.suffix.lower(),
                          "status": "error", "reason": f"worker crash: {type(e).__name__}: {e}", "flags": []}
            entries.append(entry)
            status = entry["status"]
            marker = "ok" if status == "ok" and not entry.get("flags") else status
            if done % 25 == 0 or done == total:
                print(f"[{label}] {done}/{total} (last: {marker} {entry['file']})", flush=True)
    return entries


def write_report(entries: List[dict]) -> dict:
    """Merges into any existing report (keyed by relative file path) instead
    of overwriting it, so running `--ext` in batches (small formats, then
    xls/docx, then pdf) accumulates into one complete final report covering
    the whole corpus rather than only the most recent batch."""
    if REPORT_PATH.is_file():
        try:
            prior = json.loads(REPORT_PATH.read_text(encoding="utf-8")).get("entries", [])
        except Exception:
            prior = []
    else:
        prior = []
    merged = {e["file"]: e for e in prior}
    for e in entries:
        merged[e["file"]] = e
    entries = list(merged.values())

    by_status = Counter(e["status"] for e in entries)
    by_ext = defaultdict(lambda: Counter())
    flag_counts = Counter()
    for e in entries:
        by_ext[e["ext"]][e["status"]] += 1
        for flag in e.get("flags", []):
            flag_counts[flag["check"]] += 1

    report = {
        "total": len(entries),
        "by_status": dict(by_status),
        "by_ext": {ext: dict(c) for ext, c in by_ext.items()},
        "flag_counts": dict(flag_counts),
        "entries": entries,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run(only_exts: List[str] = None) -> dict:
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    all_files = discover_files()
    if only_exts is not None:
        all_files = [f for f in all_files if f.suffix.lower() in only_exts]

    by_ext = defaultdict(list)
    for p in all_files:
        by_ext[p.suffix.lower()].append(p)

    print(f"[run] {len(all_files)} files across {len(by_ext)} extension(s): "
          f"{ {k: len(v) for k, v in sorted(by_ext.items())} }")

    entries: List[dict] = []

    doc_files = by_ext.pop(".doc", []) + by_ext.pop(".docm", [])
    if doc_files:
        staging = Path(tempfile.mkdtemp(prefix="docconv_"))
        try:
            with doc_legacy.WordConverter() as converter:
                print(f"[doc] converting {len(doc_files)} legacy .doc/.docm file(s) via Word COM (serial)...")
                entries.extend(process_doc_batch(doc_files, converter, staging))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    generic_files = []
    for ext in (".docx", ".pptx", ".xlsx", ".xls", ".gif"):
        generic_files.extend(by_ext.pop(ext, []))
    if generic_files:
        entries.extend(_run_pool(ThreadPoolExecutor, process_generic, generic_files, GENERIC_WORKERS, "generic"))

    pdf_files = by_ext.pop(".pdf", [])
    if pdf_files:
        entries.extend(_run_pool(ProcessPoolExecutor, process_pdf, pdf_files, PDF_WORKERS, "pdf"))

    for ext, files in by_ext.items():
        for f in files:
            rel = f.relative_to(INPUT_ROOT)
            entries.append({"file": str(rel), "ext": ext, "status": "skipped",
                             "reason": "no parser for extension", "flags": []})

    report = write_report(entries)
    print(f"\n[run] done: {report['by_status']}")
    print(f"[run] flags: {report['flag_counts']}")
    return report
