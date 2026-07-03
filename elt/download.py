"""
Entry point: parallel download of the public Yandex.Disk folder into input_docs/.

Usage:
    python -m elt.download
    python -m elt.download --workers 20 --dest input_docs
"""
import argparse
import time
from pathlib import Path

from .config import DOWNLOAD_WORKERS, INPUT_DIR, LIST_WORKERS, YANDEX_PUBLIC_KEY
from .yandex_downloader import ParallelYandexDiskDownloader


def main():
    ap = argparse.ArgumentParser(description="Parallel Yandex.Disk downloader")
    ap.add_argument("--dest", default=str(INPUT_DIR), help="Destination folder")
    ap.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS, help="Concurrent file downloads")
    ap.add_argument("--list-workers", type=int, default=LIST_WORKERS, help="Concurrent folder listings")
    args = ap.parse_args()

    dest = Path(args.dest)
    downloader = ParallelYandexDiskDownloader(
        YANDEX_PUBLIC_KEY, dest,
        download_workers=args.workers,
        list_workers=args.list_workers,
    )

    t0 = time.perf_counter()
    downloaded = downloader.download_all()
    elapsed = time.perf_counter() - t0

    print(f"\nDownloaded {len(downloaded)} new file(s) into {dest} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
