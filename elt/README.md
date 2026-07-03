# elt/ — Stage 1: Download

Downloads the hackathon's source corpus (a public Yandex.Disk folder, ~170 nested
category/year folders, several GB) onto local disk, fast and safely re-runnable.

## What it does

- **`yandex_downloader.py` (`ParallelYandexDiskDownloader`)** — walks the remote
  tree breadth-first, listing sibling folders **concurrently**
  (`ThreadPoolExecutor`), then downloads files in parallel
  (`DOWNLOAD_WORKERS` concurrent workers).
- **Mirrors the remote folder structure locally** — each file lands at the same
  nested path it has on the Yandex.Disk source, under `input_docs/`. This matters:
  ~50 filenames on this source repeat across *different* remote folders, so
  flattening everything into one folder (as a naive downloader would) silently
  overwrites files. Mirroring the real path is the fix.
- **Safe to interrupt / re-run** — every download writes to a `.part` sibling file
  and atomically renames on success, so a killed process leaves an inert `.part`,
  never a corrupt "real" file. Re-running skips anything already on disk at the
  size the API reports, so it's cheap to resume.
- **`reorganize_existing()`** runs automatically at the start of a download: if
  files were previously pulled down flat (e.g. by an earlier/legacy run), it moves
  them into their correct nested location by matching size, instead of
  re-downloading multiple GB that's already local. Truly ambiguous cases (same
  name *and* same size in two different remote folders) are left alone and logged.
- **`extract_archives.py`** — some source folders are uploaded as `.rar`/`.zip`
  archives, including multi-volume RAR sets (`name.part1.rar`, `name.part2.rar`, ...).
  Each archive is integrity-tested, extracted into a hidden staging folder, and only
  then swapped in for the archive (which is deleted) once 7-Zip confirms both steps
  succeeded and at least one file came out. Any failure leaves the original archive
  untouched. Runs in passes to catch archives-within-archives.

## Requirements

- `requests`
- A **RAR5-capable 7-Zip** (this source's RAR archives are RAR5; an old bundled
  7-Zip build only reads RAR3/4 and fails without a clear error). `config.py`'s
  `SEVEN_ZIP_CANDIDATES` prefers `C:\Program Files\7-Zip\7z.exe` (v22.01+) before
  falling back to `7z` on `PATH`, and raises a clear error if neither qualifies.

## How to run

```bash
python -m elt.download            # discover + download everything, in parallel
python -m elt.extract_archives    # unpack any .rar/.zip pulled down, in place
```

Both commands are idempotent — safe to re-run after a partial/interrupted run.

Tunables (worker counts, timeouts, paths) live in `config.py`.

## Output

Files land in `input_docs/` (not included in this repo — the corpus is several GB;
see the top-level README for why data isn't checked in). Run the commands above
against a real Yandex.Disk public link to regenerate it locally.

## Status

Done and working. This is a clean rewrite of the download stage only — it does not
parse/convert anything (see `../parsing/` for that).
