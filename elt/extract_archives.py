"""
Extracts .rar and .zip archives found under input_docs/ **in place**: each
archive is replaced by a folder of the same stem containing its contents.
This is destructive (archives are deleted after extraction), so every step
is conservative — an archive is only ever removed after its extraction is
independently verified to have succeeded:

  1. integrity test (`7z t`) before touching anything
  2. extract into a hidden staging folder, never directly into the final name
  3. only rename staging -> final and delete the source archive(s) once
     7-Zip reported success ("Everything is Ok") on both the test and the
     extract step, and at least one file came out
  4. any failure at any step leaves the original archive(s) untouched and
     is reported, never guessed past

Two distinct multi-volume namings are both treated as one unit each:
extraction reads from the lowest-numbered volume (7-Zip pulls in the rest
automatically by scanning the folder) and, on success, every volume in the
set is deleted.
  - RAR multi-volume: `name.part1.rar`, `name.part2.rar`, ...
  - Raw split volumes: `name.zip.001`, `name.zip.002`, ... (7-Zip's `-v`
    split format — the joined bytes are themselves a plain archive, e.g.
    `Type = Split` wrapping a `.zip`). The trailing `.NNN` is a numeric
    extension, not a real one, so these are invisible to a plain
    `suffix in {'.zip', '.rar'}` check — they have to be matched by regex
    on the full filename instead.
Neither is the downloader's own `.part` atomic-write suffix
(elt/yandex_downloader.py), which is always appended after the *full*
filename (e.g. `foo.rar.part`) and never persists past a successful
download.

Runs in a loop: after each pass, re-scans for archives newly revealed
inside just-extracted folders (an archive can contain another archive),
stopping once a pass makes no further progress.
"""
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from .config import EXTRACT_WORKERS, INPUT_DIR, SEVEN_ZIP_CANDIDATES

_RAR_PART_RE = re.compile(r"\.part(\d+)\.rar$", re.IGNORECASE)
_SPLIT_RE = re.compile(r"\.(zip|rar|7z)\.(\d+)$", re.IGNORECASE)
_ARCHIVE_EXTS = (".rar", ".zip", ".7z")


def find_seven_zip() -> str:
    for candidate in SEVEN_ZIP_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(
        "No 7-Zip binary found. Tried: " + ", ".join(SEVEN_ZIP_CANDIDATES) +
        ". Install 7-Zip (needs RAR5 support, 21.x+) or add it to PATH."
    )


def discover_archive_sets(root: Path) -> List[Tuple[Path, List[Path]]]:
    """Return (entry_point, all_volumes) pairs: one per plain .zip/.rar/.7z
    file, one per multi-volume RAR set, and one per raw split-volume set
    (entry_point = lowest-numbered volume, all_volumes = every volume, so
    callers can delete the whole set together)."""
    all_files = [p for p in root.rglob("*") if p.is_file()]

    rar_multipart = {}    # (parent, base) -> {part_num: path}
    split_multipart = {}  # (parent, base) -> {vol_num: path}
    singles = []
    for p in all_files:
        m_split = _SPLIT_RE.search(p.name)
        if m_split:
            base = p.name[: m_split.start()] + "." + m_split.group(1)
            split_multipart.setdefault((p.parent, base), {})[int(m_split.group(2))] = p
            continue
        if p.suffix.lower() not in _ARCHIVE_EXTS:
            continue
        m_rar = _RAR_PART_RE.search(p.name)
        if m_rar:
            base = p.name[: m_rar.start()]
            rar_multipart.setdefault((p.parent, base), {})[int(m_rar.group(1))] = p
        else:
            singles.append(p)

    sets = [(p, [p]) for p in singles]
    for _key, parts in rar_multipart.items():
        ordered = [parts[k] for k in sorted(parts)]
        sets.append((ordered[0], ordered))
    for _key, parts in split_multipart.items():
        ordered = [parts[k] for k in sorted(parts)]
        sets.append((ordered[0], ordered))
    return sorted(sets, key=lambda s: s[0])


def _target_dir_for(entry_point: Path) -> Path:
    m_rar = _RAR_PART_RE.search(entry_point.name)
    if m_rar:
        stem = entry_point.name[: m_rar.start()]
    else:
        m_split = _SPLIT_RE.search(entry_point.name)
        stem = entry_point.name[: m_split.start()] if m_split else entry_point.stem
    return entry_point.parent / stem


def _run_7z(seven_zip: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [seven_zip, *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _extract_one(seven_zip: str, entry_point: Path, volumes: List[Path]) -> Optional[str]:
    target_dir = _target_dir_for(entry_point)

    if target_dir.is_dir() and any(target_dir.iterdir()):
        # Already extracted (this run or an earlier one) — trust that prior,
        # verified extraction and just clear out the now-redundant archive(s).
        for v in volumes:
            v.unlink(missing_ok=True)
        return None
    if target_dir.exists() and not target_dir.is_dir():
        return f"{entry_point}: target path {target_dir} exists and is not a folder — skipped"

    test = _run_7z(seven_zip, "t", str(entry_point))
    if test.returncode != 0 or "Everything is Ok" not in test.stdout:
        return f"{entry_point}: integrity test failed (rc={test.returncode})"

    staging = entry_point.parent / f".extracting_{target_dir.name}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    extract = _run_7z(seven_zip, "x", str(entry_point), "-y", f"-o{staging}")
    if extract.returncode != 0 or "Everything is Ok" not in extract.stdout:
        shutil.rmtree(staging, ignore_errors=True)
        return f"{entry_point}: extraction failed (rc={extract.returncode}): {extract.stderr.strip()[:300]}"

    if not any(staging.rglob("*")):
        shutil.rmtree(staging, ignore_errors=True)
        return f"{entry_point}: extraction produced no files"

    staging.replace(target_dir)
    for v in volumes:
        v.unlink()
    return None


def extract_all(root: Path = INPUT_DIR, workers: int = EXTRACT_WORKERS, max_passes: int = 5) -> None:
    seven_zip = find_seven_zip()
    print(f"[extract] using 7-Zip at {seven_zip}")

    total_ok = 0
    total_failed = 0
    for pass_num in range(1, max_passes + 1):
        sets = discover_archive_sets(root)
        if not sets:
            print(f"[extract] pass {pass_num}: no archives left")
            break
        print(f"[extract] pass {pass_num}: {len(sets)} archive(s) found")

        pass_errors = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_one, seven_zip, ep, vols): ep for ep, vols in sets}
            for fut in as_completed(futures):
                ep = futures[fut]
                done += 1
                err = fut.result()
                if err:
                    pass_errors.append(err)
                    print(f"[{done}/{len(sets)}] FAILED: {err}")
                else:
                    print(f"[{done}/{len(sets)}] ok: {ep.relative_to(root)}")

        total_ok += len(sets) - len(pass_errors)
        total_failed_this_pass = len(pass_errors)
        if total_failed_this_pass == len(sets):
            total_failed += total_failed_this_pass
            print(f"[extract] no progress this pass — stopping")
            break
        total_failed = total_failed_this_pass  # only the last pass's failures matter going forward

    remaining = discover_archive_sets(root)
    print(f"\n[extract] done: {total_ok} extracted+removed, {len(remaining)} archive(s) still present (see failures above)")


if __name__ == "__main__":
    extract_all()
