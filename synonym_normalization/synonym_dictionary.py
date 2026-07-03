import json
from pathlib import Path
from typing import Dict, Set

from .canonicalizer import canonicalize_text


class SynonymDictionary:
    """
    Stores aliases and resolves extracted names to canonical names.
    Can be loaded from a JSON dictionary or extended from entity mentions.
    """

    def __init__(self) -> None:
        self._canonical_map: Dict[str, str] = {}
        self._known_aliases: Dict[str, Set[str]] = {}

    def load(self, filepath: Path) -> None:
        if not filepath.exists():
            return

        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for canonical, aliases in data.items():
                self.add_term(canonical, aliases)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {k: sorted(v) for k, v in self._known_aliases.items()}
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_term(self, canonical: str, aliases: list[str]) -> None:
        canonical_key = canonicalize_text(canonical)
        if canonical_key not in self._known_aliases:
            self._known_aliases[canonical_key] = set()

        self._known_aliases[canonical_key].add(canonical)
        self._canonical_map[canonical_key] = canonical

        for alias in aliases:
            self._known_aliases[canonical_key].add(alias)
            alias_key = canonicalize_text(alias)
            self._canonical_map[alias_key] = canonical

    def resolve(self, text: str) -> str:
        """Return a canonical name if a known alias matches; otherwise return input."""
        text_key = canonicalize_text(text)
        return self._canonical_map.get(text_key, text)
