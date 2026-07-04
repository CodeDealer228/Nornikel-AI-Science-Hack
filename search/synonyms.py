from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import yaml


WORD_BOUNDARY_CHARS = r"A-Za-zА-Яа-яЁё0-9"


SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
)


@dataclass(frozen=True)
class SynonymGroup:
    canonical_id: str
    type: str
    canonical_name: str
    aliases: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]


def normalize_for_match(text: str) -> str:
    text = text.translate(SUBSCRIPT_TRANSLATION)
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-").replace("‒", "-")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ordered_unique(items: list[str]) -> list[str]:
    result = []
    seen = set()

    for item in items:
        if not item:
            continue

        key = item.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def make_alias_pattern(alias: str) -> re.Pattern[str]:
    alias_norm = normalize_for_match(alias)

    escaped = re.escape(alias_norm)
    escaped = escaped.replace(r"\ ", r"\s+")

    left_boundary = ""
    right_boundary = ""

    if alias_norm and re.match(rf"[{WORD_BOUNDARY_CHARS}]", alias_norm[0]):
        left_boundary = rf"(?<![{WORD_BOUNDARY_CHARS}])"

    if alias_norm and re.match(rf"[{WORD_BOUNDARY_CHARS}]", alias_norm[-1]):
        right_boundary = rf"(?![{WORD_BOUNDARY_CHARS}])"

    return re.compile(left_boundary + escaped + right_boundary, flags=re.IGNORECASE)


class SynonymExpander:
    def __init__(self, groups: list[SynonymGroup]) -> None:
        self.groups = groups

    @classmethod
    def empty(cls) -> "SynonymExpander":
        return cls(groups=[])

    @classmethod
    def from_yaml(cls, path: Path) -> "SynonymExpander":
        if not path.exists():
            return cls.empty()

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        raw_groups = data.get("groups", [])
        groups: list[SynonymGroup] = []

        for raw_group in raw_groups:
            canonical_id = str(raw_group["canonical_id"])
            group_type = str(raw_group.get("type", "Unknown"))
            canonical_name = str(raw_group["canonical_name"])
            aliases = tuple(str(alias) for alias in raw_group.get("aliases", []))

            searchable_aliases = ordered_unique([canonical_name, *aliases])
            patterns = tuple(make_alias_pattern(alias) for alias in searchable_aliases)

            groups.append(
                SynonymGroup(
                    canonical_id=canonical_id,
                    type=group_type,
                    canonical_name=canonical_name,
                    aliases=aliases,
                    patterns=patterns,
                )
            )

        return cls(groups=groups)

    @classmethod
    def from_project(cls, path: Path | None = None) -> "SynonymExpander":
        if path is not None:
            return cls.from_yaml(path)

        # Поддержим оба варианта, потому что легко ошибиться в названии.
        candidates = [
            Path("resources/synonyms.yaml"),
            Path("resources/synonims.yaml"),
        ]

        for candidate in candidates:
            if candidate.exists():
                return cls.from_yaml(candidate)

        return cls.empty()

    def find_matches(self, query: str) -> list[dict[str, Any]]:
        query_norm = normalize_for_match(query)

        matches = []

        for group in self.groups:
            matched_aliases = []

            for alias in [group.canonical_name, *group.aliases]:
                pattern = make_alias_pattern(alias)

                if pattern.search(query_norm):
                    matched_aliases.append(alias)

            if matched_aliases:
                matches.append(
                    {
                        "canonical_id": group.canonical_id,
                        "type": group.type,
                        "canonical_name": group.canonical_name,
                        "matched_aliases": ordered_unique(matched_aliases),
                    }
                )

        return matches

    def expand_query_tokens(
        self,
        *,
        query: str,
        base_tokens: list[str],
        tokenizer: Callable[[str], list[str]],
        include_canonical_id: bool = False,
        max_groups: int = 12,
    ) -> list[str]:
        """
        Расширяет только lexical/BM25 query.

        include_canonical_id=False для query-only expansion.
        canonical_id имеет смысл добавлять только если ты также добавляешь canonical_id
        в документы при построении BM25-индекса.
        """
        expanded = list(base_tokens)

        matches = self.find_matches(query)

        for match in matches[:max_groups]:
            canonical_name = match["canonical_name"]
            canonical_id = match["canonical_id"]

            expanded.extend(tokenizer(canonical_name))

            group = self._get_group_by_id(canonical_id)
            if group is None:
                continue

            for alias in group.aliases:
                expanded.extend(tokenizer(alias))

            if include_canonical_id:
                expanded.append(canonical_id)

        return ordered_unique(expanded)

    def _get_group_by_id(self, canonical_id: str) -> SynonymGroup | None:
        for group in self.groups:
            if group.canonical_id == canonical_id:
                return group

        return None