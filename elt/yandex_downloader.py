"""
Parallel downloader for public Yandex.Disk folders.

Replaces the sequential download in a.py's YandexDiskDownloader, which lists
and fetches one file at a time into a single flat folder. This version:
  - mirrors the remote folder structure under dest_dir instead of flattening
    everything (the source has ~170 nested category/year folders, and ~50
    filenames repeat across different folders — flattening silently drops
    all but the last-downloaded copy of each repeated name)
  - lists sibling subfolders concurrently instead of walking them one by one
  - downloads files concurrently with a thread pool (I/O-bound, so threads
    are sufficient — no need for asyncio)
  - skips files already present locally with a matching size, so re-running
    only fetches what's missing
  - writes to a .part file and atomically renames on completion, so a
    crash mid-download can't leave a corrupt file that looks "already there"
"""
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests

from .config import (
    CHUNK_SIZE,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_WORKERS,
    LIST_WORKERS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    YANDEX_API_BASE,
)

_print_lock = threading.Lock()

# Windows forbids these in path components; Yandex item names occasionally
# contain them (mostly ':' in report titles). Replace with '_'.
_FORBIDDEN_CHARS = '<>:"|?*'


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _sanitize_component(name: str) -> str:
    for ch in _FORBIDDEN_CHARS:
        name = name.replace(ch, "_")
    return name.rstrip(" .") or "_"


def relative_path(item: dict) -> Path:
    """Local relative path for a remote item, mirroring its folder structure."""
    remote_path = item.get("path", "").replace("disk:", "").lstrip("/")
    parts = [p for p in remote_path.split("/") if p]
    if not parts:
        parts = [item.get("name", "unnamed")]
    return Path(*[_sanitize_component(p) for p in parts])


class ParallelYandexDiskDownloader:
    """Downloads all files from a public Yandex.Disk folder in parallel,
    mirroring the remote folder structure under dest_dir."""

    def __init__(
        self,
        public_key: str,
        dest_dir: Path,
        download_workers: int = DOWNLOAD_WORKERS,
        list_workers: int = LIST_WORKERS,
    ):
        self.public_key = public_key
        self.dest_dir = dest_dir
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.download_workers = download_workers
        self.list_workers = list_workers
        self._session = requests.Session()

    # ---- listing ------------------------------------------------------
    def _list_dir_page(self, path: str, offset: int) -> Dict[str, Any]:
        params = {
            "public_key": self.public_key,
            "path": path,
            "offset": offset,
            "limit": 200,
        }
        r = self._session.get(YANDEX_API_BASE, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _list_dir(self, path: str) -> List[dict]:
        """List all items in one folder, paginating sequentially (each page
        depends on the previous one's size, so pagination itself isn't
        parallelized — only sibling folders are)."""
        items: List[dict] = []
        offset = 0
        while True:
            data = self._list_dir_page(path, offset)
            batch = data.get("_embedded", {}).get("items", [])
            items.extend(batch)
            if len(batch) < 200:
                break
            offset += 200
        return items

    def discover_files(self, path: str = "/") -> List[dict]:
        """Recursively discover every file, listing folders breadth-first
        with up to `list_workers` folders in flight at once."""
        files: List[dict] = []
        dirs_to_scan = [path]
        with ThreadPoolExecutor(max_workers=self.list_workers) as pool:
            while dirs_to_scan:
                futures = {pool.submit(self._list_dir, d): d for d in dirs_to_scan}
                dirs_to_scan = []
                for fut in as_completed(futures):
                    for item in fut.result():
                        if item.get("type") == "file":
                            files.append(item)
                        elif item.get("type") == "dir":
                            dirs_to_scan.append(item.get("path", "").replace("disk:", ""))
        return files

    # ---- reorganizing pre-existing flat downloads ----------------------
    def reorganize_existing(self, remote_files: List[dict]) -> Dict[str, int]:
        """Move files that were previously downloaded flat (directly into
        dest_dir, by a.py or an earlier version of this downloader) into
        their correct nested location, instead of re-downloading them.

        Only moves a flat file when its basename maps to exactly one remote
        path (no ambiguity) or, for a basename that maps to several remote
        paths, when exactly one of them matches the flat file's size.
        Anything left ambiguous is reported, not touched — it will simply
        be (re)downloaded fresh into its correct nested location.
        """
        by_name: Dict[str, List[dict]] = defaultdict(list)
        for item in remote_files:
            by_name[item.get("name", "")].append(item)

        moved = 0
        skipped_ambiguous = 0
        for name, candidates in by_name.items():
            flat_path = self.dest_dir / name
            if not flat_path.is_file():
                continue
            flat_size = flat_path.stat().st_size

            if len(candidates) == 1:
                target_candidates = candidates
            else:
                target_candidates = [c for c in candidates if c.get("size") == flat_size]
                if len(target_candidates) != 1:
                    skipped_ambiguous += 1
                    _log(
                        f"[reorganize] ambiguous: {name!r} matches "
                        f"{len(candidates)} remote paths, {len(target_candidates)} by size — left in place"
                    )
                    continue

            target = self.dest_dir / relative_path(target_candidates[0])
            if target == flat_path:
                continue
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            flat_path.rename(target)
            moved += 1

        _log(f"[reorganize] moved {moved} file(s) into nested folders, {skipped_ambiguous} left ambiguous")
        return {"moved": moved, "ambiguous": skipped_ambiguous}

    # ---- downloading ----------------------------------------------------
    def _needs_download(self, item: dict) -> bool:
        dest = self.dest_dir / relative_path(item)
        if not dest.exists():
            return True
        size = item.get("size")
        return size is not None and dest.stat().st_size != size

    def _download_one(self, item: dict) -> Path:
        url = item["file"]
        dest = self.dest_dir / relative_path(item)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")

        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with self._session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                tmp.replace(dest)
                return dest
            except Exception as e:
                last_err = e
                tmp.unlink(missing_ok=True)
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_err}")

    def download_all(self, path: str = "/") -> List[Path]:
        _log(f"[list] discovering files under {path!r} ...")
        all_files = self.discover_files(path)
        _log(f"[list] found {len(all_files)} files on disk")

        self.reorganize_existing(all_files)

        to_fetch = [f for f in all_files if f.get("file") and self._needs_download(f)]
        already_present = len(all_files) - len(to_fetch)
        _log(
            f"[skip] {already_present} already present in {self.dest_dir} "
            f"(matched by size), {len(to_fetch)} to download"
        )

        downloaded: List[Path] = []
        if not to_fetch:
            return downloaded

        done = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=self.download_workers) as pool:
            futures = {pool.submit(self._download_one, item): item for item in to_fetch}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    p = fut.result()
                    downloaded.append(p)
                    done += 1
                    _log(f"[{done + failed}/{len(to_fetch)}] ok: {p.relative_to(self.dest_dir)}")
                except Exception as e:
                    failed += 1
                    _log(f"[{done + failed}/{len(to_fetch)}] FAILED: {item.get('name')}: {e}")

        if failed:
            _log(f"[warn] {failed} file(s) failed to download, see log above")
        return downloaded
