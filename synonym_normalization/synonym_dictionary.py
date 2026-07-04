"""Loads the human-curated ../resources/synonyms.yaml (ported from the
search-module-update branch, see README.md) into an O(1) lookup: normalized
surface form -> canonical group. This is the only *automatic* cross-document
merge source this package trusts -- every group in that file was promoted
there by a human after appearing in resources/synonym_candidates.yaml, so
resolving through it is not a guess.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from canonicalizer import normalize


@dataclass(frozen=True)
class SynonymGroup:
    canonical_id: str
    type: str
    canonical_name: str
    aliases: tuple


class SynonymDictionary:
    def __init__(self, groups: list):
        self.groups = groups
        self._by_normalized_text: dict = {}
        for group in groups:
            for surface in (group.canonical_name, *group.aliases):
                self._by_normalized_text.setdefault(normalize(surface), group)

    @classmethod
    def load(cls, path: Path) -> "SynonymDictionary":
        if not path.exists():
            return cls([])
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        groups = []
        for raw in data.get("groups", []):
            groups.append(SynonymGroup(
                canonical_id=str(raw["canonical_id"]),
                type=str(raw.get("type", "Unknown")),
                canonical_name=str(raw["canonical_name"]),
                aliases=tuple(str(a) for a in raw.get("aliases", [])),
            ))
        return cls(groups)

    def resolve(self, text: str) -> Optional[SynonymGroup]:
        """Return the curated group this surface form belongs to, if any."""
        return self._by_normalized_text.get(normalize(text))
