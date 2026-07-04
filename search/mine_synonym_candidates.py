from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ABBR_RE = re.compile(
    r"(?P<full>[А-Яа-яA-Za-zЁё][А-Яа-яA-Za-zЁё0-9\s,\-]{6,80})"
    r"\s*\("
    r"(?P<abbr>[A-ZА-ЯЁ]{2,12}|[A-ZА-ЯЁ][A-ZА-ЯЁ0-9\-]{1,15})"
    r"\)",
)

SLASH_PAIR_RE = re.compile(
    r"(?P<left>[А-Яа-яЁё][А-Яа-яЁё0-9\s,\-]{3,60})"
    r"\s*/\s*"
    r"(?P<right>[A-Za-z][A-Za-z0-9\s,\-]{3,60})"
)

DASH_DEFINITION_RE = re.compile(
    r"(?P<abbr>[A-ZА-ЯЁ]{2,12})"
    r"\s*[—–-]\s*"
    r"(?P<full>[А-Яа-яA-Za-zЁё][А-Яа-яA-Za-zЁё0-9\s,\-]{6,80})"
)

NOISE_WORDS = {
    "таблица",
    "рисунок",
    "figure",
    "table",
    "образец",
    "sample",
    "приложение",
    "источник",
}


def normalize_phrase(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,.;:()[]{}")
    return text


def looks_noisy(text: str) -> bool:
    low = text.lower()

    if len(low) < 2:
        return True

    if any(word in low for word in NOISE_WORDS):
        return True

    # Слишком много цифр — часто номер, год, ссылка.
    digits = sum(ch.isdigit() for ch in text)
    if digits > max(3, len(text) // 3):
        return True

    return False


def add_candidate(
    candidates: dict[tuple[str, str], dict[str, Any]],
    *,
    source_type: str,
    alias_a: str,
    alias_b: str,
    chunk_id: str,
    title: str | None,
    context: str,
) -> None:
    alias_a = normalize_phrase(alias_a)
    alias_b = normalize_phrase(alias_b)

    if looks_noisy(alias_a) or looks_noisy(alias_b):
        return

    # Чтобы одинаковые пары не плодились в разном порядке.
    key = tuple(sorted([alias_a.lower(), alias_b.lower()]))

    if key not in candidates:
        candidates[key] = {
            "aliases": sorted({alias_a, alias_b}),
            "source_type": source_type,
            "count": 0,
            "examples": [],
        }

    candidates[key]["count"] += 1

    if len(candidates[key]["examples"]) < 3:
        candidates[key]["examples"].append(
            {
                "chunk_id": chunk_id,
                "title": title,
                "context": context[:300],
            }
        )


def iter_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            chunks.append(json.loads(line))

    return chunks


def get_chunk_text(chunk: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk_text"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def get_chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or "unknown")


def get_title(chunk: dict[str, Any]) -> str | None:
    value = chunk.get("title") or chunk.get("source_document")
    if value:
        return str(value)
    return None


def mine_candidates(chunks_path: Path) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    for chunk in iter_chunks(chunks_path):
        text = get_chunk_text(chunk)
        chunk_id = get_chunk_id(chunk)
        title = get_title(chunk)

        if not text:
            continue

        for match in ABBR_RE.finditer(text):
            full = match.group("full")
            abbr = match.group("abbr")

            context_start = max(0, match.start() - 80)
            context_end = min(len(text), match.end() + 80)

            add_candidate(
                candidates,
                source_type="parentheses_abbreviation",
                alias_a=full,
                alias_b=abbr,
                chunk_id=chunk_id,
                title=title,
                context=text[context_start:context_end],
            )

        for match in SLASH_PAIR_RE.finditer(text):
            left = match.group("left")
            right = match.group("right")

            context_start = max(0, match.start() - 80)
            context_end = min(len(text), match.end() + 80)

            add_candidate(
                candidates,
                source_type="slash_pair",
                alias_a=left,
                alias_b=right,
                chunk_id=chunk_id,
                title=title,
                context=text[context_start:context_end],
            )

        for match in DASH_DEFINITION_RE.finditer(text):
            abbr = match.group("abbr")
            full = match.group("full")

            context_start = max(0, match.start() - 80)
            context_end = min(len(text), match.end() + 80)

            add_candidate(
                candidates,
                source_type="dash_definition",
                alias_a=full,
                alias_b=abbr,
                chunk_id=chunk_id,
                title=title,
                context=text[context_start:context_end],
            )

    result = list(candidates.values())
    result.sort(key=lambda item: item["count"], reverse=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("parsed_data/chunks.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("resources/synonym_candidates.yaml"),
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    candidates = mine_candidates(args.chunks)
    candidates = [item for item in candidates if item["count"] >= args.min_count]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {"candidates": candidates},
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"Saved {len(candidates)} candidates to {args.out}")


if __name__ == "__main__":
    main()