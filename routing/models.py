from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ExtractionRoute(StrEnum):
    NATASHA_ONLY = "natasha_only"
    LLM_ONLY = "llm_only"
    ENSEMBLE = "ensemble"
    SKIP = "skip"


@dataclass(frozen=True)
class RoutingSignal:
    name: str
    value: float | str | bool
    weight: float = 1.0


@dataclass(frozen=True)
class RoutingDecision:
    route: ExtractionRoute
    confidence: float
    reasons: tuple[str, ...] = ()
    signals: tuple[RoutingSignal, ...] = field(default_factory=tuple)
