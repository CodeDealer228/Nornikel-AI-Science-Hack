from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

@dataclass
class Sentence:
    text: str
    start: int
    stop: int

@runtime_checkable
class SentenceSegmenter(Protocol):
    def segment(self, text: str) -> List[Sentence]: ...
