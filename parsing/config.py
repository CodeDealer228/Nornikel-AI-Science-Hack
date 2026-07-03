"""Configuration for the deterministic parsing pipeline (parsing/)."""
from pathlib import Path

INPUT_ROOT = Path("input_docs") / "Источники информации"
OUTPUT_ROOT = Path("parsed_data")
TEXTS_DIR = OUTPUT_ROOT / "texts"
IMAGES_DIR = OUTPUT_ROOT / "images"
REPORT_PATH = OUTPUT_ROOT / "parse_report.json"

# --- PDF image extraction (pymupdf4llm) ---
PDF_IMAGE_FORMAT = "png"
PDF_IGNORE_GRAPHICS = True
PDF_IMAGE_SIZE_LIMIT = 0.08  # fraction of page area, pymupdf4llm's own filter
PDF_IMAGE_MIN_BYTES = 3072  # post-filter: drop page-furniture/icon junk below this

# --- validation thresholds ---
LOW_YIELD_CHARS_PER_UNIT = 40  # min avg chars per page/slide/sheet before flagging
ENCODING_SUSPECT_RATIO = 0.02  # fraction of replacement/control chars before flagging

# --- concurrency ---
PDF_WORKERS = 10     # ProcessPoolExecutor, CPU-bound (fitz text/table/image extraction); machine has 12 cores
GENERIC_WORKERS = 8  # ThreadPoolExecutor, for docx/xlsx/pptx/xls/image (I/O + C-ext)

# --- known file-signature bytes ---
SIGNATURES = {
    ".pdf": b"%PDF",
    ".docx": b"PK\x03\x04",
    ".docm": b"PK\x03\x04",
    ".pptx": b"PK\x03\x04",
    ".xlsx": b"PK\x03\x04",
    ".doc": b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    ".xls": b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    ".gif": b"GIF8",
}

IMAGE_SIGNATURES = {
    b"BM": "bmp",
    b"\x89PNG": "png",
    b"\xff\xd8": "jpg",
    b"GIF8": "gif",
}
