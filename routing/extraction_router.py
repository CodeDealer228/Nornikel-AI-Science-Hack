from __future__ import annotations

import re
from collections.abc import Sequence

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation

from .models import ExtractionRoute, RoutingDecision, RoutingSignal

_TECHNICAL_VALUE_RE = re.compile(r"\d+(?:[,.]\d+)?\s*(?:%|°C|г/л|мг/л|м3/ч|А/м2|pH)", re.I)
_RELATION_CUE_RE = re.compile(
    r"\b(?:влияет|повышает|снижает|использует|получен|проводили|зависит|"
    r"подтвержден|описан|применяется|заменяет)\b",
    re.I,
)


class ExtractionRouter:
    """
    Chooses an extraction path per chunk using cheap deterministic signals and
    optional extractor feedback from previous attempts.
    """

    def __init__(
        self,
        short_text_chars: int = 350,
        long_text_chars: int = 4500,
        low_confidence_threshold: float = 0.55,
        sparse_coverage_threshold: float = 0.003,
    ) -> None:
        self.short_text_chars = short_text_chars
        self.long_text_chars = long_text_chars
        self.low_confidence_threshold = low_confidence_threshold
        self.sparse_coverage_threshold = sparse_coverage_threshold

    def route_chunk(
        self,
        text: str,
        natasha_entities: Sequence[EnrichedEntity] | None = None,
        llm_entities: Sequence[EnrichedEntity] | None = None,
        llm_relations: Sequence[EnrichedRelation] | None = None,
    ) -> RoutingDecision:
        text = text or ""
        natasha_entities = natasha_entities or []
        llm_entities = llm_entities or []
        llm_relations = llm_relations or []

        if not text.strip():
            return RoutingDecision(
                route=ExtractionRoute.SKIP,
                confidence=1.0,
                reasons=("empty_chunk",),
            )

        signals = self._collect_signals(text, natasha_entities, llm_entities, llm_relations)
        reasons: list[str] = []

        if len(text) <= self.short_text_chars and not _RELATION_CUE_RE.search(text):
            reasons.append("short_low_relation_signal")
            return RoutingDecision(
                route=ExtractionRoute.NATASHA_ONLY,
                confidence=0.76,
                reasons=tuple(reasons),
                signals=tuple(signals),
            )

        if len(text) >= self.long_text_chars:
            reasons.append("long_chunk_requires_llm_semantics")
            return RoutingDecision(
                route=ExtractionRoute.ENSEMBLE,
                confidence=0.82,
                reasons=tuple(reasons),
                signals=tuple(signals),
            )

        technical_density = self._technical_density(text)
        relation_cues = len(_RELATION_CUE_RE.findall(text))
        natasha_coverage = self._coverage(natasha_entities, len(text))
        llm_confidence = self._average_confidence(llm_entities, llm_relations)

        if technical_density > 0.0015 or relation_cues >= 2:
            reasons.append("technical_or_relation_dense")
            if natasha_entities:
                reasons.append("natasha_seeds_available")
                route = ExtractionRoute.ENSEMBLE
                confidence = 0.84
            else:
                route = ExtractionRoute.LLM_ONLY
                confidence = 0.75
            return RoutingDecision(route, confidence, tuple(reasons), tuple(signals))

        if llm_entities and llm_confidence < self.low_confidence_threshold:
            reasons.append("low_llm_confidence_requires_ensemble")
            return RoutingDecision(ExtractionRoute.ENSEMBLE, 0.78, tuple(reasons), tuple(signals))

        if natasha_entities and natasha_coverage < self.sparse_coverage_threshold:
            reasons.append("sparse_natasha_coverage_requires_llm")
            return RoutingDecision(ExtractionRoute.LLM_ONLY, 0.72, tuple(reasons), tuple(signals))

        if natasha_entities and not relation_cues:
            reasons.append("entity_only_chunk")
            return RoutingDecision(ExtractionRoute.NATASHA_ONLY, 0.7, tuple(reasons), tuple(signals))

        reasons.append("default_semantic_extraction")
        return RoutingDecision(ExtractionRoute.ENSEMBLE, 0.68, tuple(reasons), tuple(signals))

    def fallback_after_extraction(
        self,
        decision: RoutingDecision,
        entities: Sequence[EnrichedEntity],
        relations: Sequence[EnrichedRelation],
    ) -> RoutingDecision:
        if decision.route == ExtractionRoute.SKIP:
            return decision
        if not entities and decision.route != ExtractionRoute.LLM_ONLY:
            return RoutingDecision(
                ExtractionRoute.LLM_ONLY,
                0.74,
                decision.reasons + ("no_entities_after_primary_route",),
                decision.signals,
            )
        if entities and not relations and decision.route == ExtractionRoute.NATASHA_ONLY:
            return RoutingDecision(
                ExtractionRoute.ENSEMBLE,
                0.7,
                decision.reasons + ("entities_without_relations",),
                decision.signals,
            )
        low_conf = self._average_confidence(entities, relations) < self.low_confidence_threshold
        if low_conf and decision.route != ExtractionRoute.ENSEMBLE:
            return RoutingDecision(
                ExtractionRoute.ENSEMBLE,
                0.73,
                decision.reasons + ("low_confidence_fallback",),
                decision.signals,
            )
        return decision

    def _collect_signals(
        self,
        text: str,
        natasha_entities: Sequence[EnrichedEntity],
        llm_entities: Sequence[EnrichedEntity],
        llm_relations: Sequence[EnrichedRelation],
    ) -> list[RoutingSignal]:
        return [
            RoutingSignal("text_chars", len(text)),
            RoutingSignal("technical_density", self._technical_density(text)),
            RoutingSignal("relation_cues", len(_RELATION_CUE_RE.findall(text))),
            RoutingSignal("natasha_entity_count", len(natasha_entities)),
            RoutingSignal("llm_entity_count", len(llm_entities)),
            RoutingSignal("llm_relation_count", len(llm_relations)),
            RoutingSignal("llm_average_confidence", self._average_confidence(llm_entities, llm_relations)),
        ]

    def _technical_density(self, text: str) -> float:
        return len(_TECHNICAL_VALUE_RE.findall(text)) / max(len(text), 1)

    def _coverage(self, entities: Sequence[EnrichedEntity], text_len: int) -> float:
        span = sum(max(0, ent.char_end - ent.char_start) for ent in entities)
        return span / max(text_len, 1)

    def _average_confidence(
        self,
        entities: Sequence[EnrichedEntity],
        relations: Sequence[EnrichedRelation],
    ) -> float:
        values = [item.confidence for item in list(entities) + list(relations)]
        return sum(values) / len(values) if values else 0.0
