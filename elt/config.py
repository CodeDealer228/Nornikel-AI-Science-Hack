"""Configuration for the parallel Yandex.Disk downloader."""
from pathlib import Path

YANDEX_PUBLIC_KEY = "https://disk.yandex.ru/d/npigiuw4Rbe9Pg"
YANDEX_API_BASE = "https://cloud-api.yandex.ru/v1/disk/public/resources"

# Relative to the current working directory (run from the repo root),
# same folder a.py already downloads into and partially populated.
INPUT_DIR = Path("input_docs")

DOWNLOAD_WORKERS = 128   # concurrent file downloads
LIST_WORKERS = 64        # concurrent folder-listing calls
REQUEST_TIMEOUT = 30    # seconds, listing API calls
DOWNLOAD_TIMEOUT = 180  # seconds, per file
CHUNK_SIZE = 1 << 16
MAX_RETRIES = 3

# Candidate paths for a RAR5-capable 7-Zip binary (the one bundled with some
# other tools on this machine is a 2009 build that can't read RAR5). First
# existing path wins; falls back to `7z` on PATH.
SEVEN_ZIP_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "7z",
]
EXTRACT_WORKERS = 4
