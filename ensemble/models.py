from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation


class EvidenceSource(StrEnum):
    NATASHA = "natasha"
    LLM = "yandex_llm"
    ENSEMBLE = "ensemble"


class MergeReason(StrEnum):
    EXACT_CANONICAL_MATCH = "exact_canonical_match"
    RELATION_ENDPOINT_MATCH = "relation_endpoint_match"
    CONFLICT_RESOLVED_BY_CONFIDENCE = "conflict_resolved_by_confidence"
    SINGLE_SOURCE = "single_source"


@dataclass(frozen=True)
class EnsembleDecision:
    output_id: str
    sources: tuple[str, ...]
    confidence: float
    reason: MergeReason
    needs_review: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class EnsembleResult:
    entities: list[EnrichedEntity] = field(default_factory=list)
    relations: list[EnrichedRelation] = field(default_factory=list)
    entity_decisions: list[EnsembleDecision] = field(default_factory=list)
    relation_decisions: list[EnsembleDecision] = field(default_factory=list)
