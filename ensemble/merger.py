from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation
from synonym_normalization.canonicalizer import canonicalize_text

from .models import EnsembleDecision, EnsembleResult, EvidenceSource, MergeReason


class EnsembleMerger:
    """
    Merges independent Natasha and LLM extraction outputs into one graph-ready
    entity/relation set while preserving provenance and review flags.
    """

    def __init__(
        self,
        llm_weight: float = 0.7,
        natasha_weight: float = 0.3,
        single_source_review_threshold: float = 0.58,
        conflict_margin: float = 0.12,
    ) -> None:
        self.llm_weight = llm_weight
        self.natasha_weight = natasha_weight
        self.single_source_review_threshold = single_source_review_threshold
        self.conflict_margin = conflict_margin

    def merge(
        self,
        natasha_entities: Iterable[EnrichedEntity],
        llm_entities: Iterable[EnrichedEntity],
        natasha_relations: Iterable[EnrichedRelation] | None = None,
        llm_relations: Iterable[EnrichedRelation] | None = None,
    ) -> EnsembleResult:
        entity_result, entity_id_map = self.merge_entities(natasha_entities, llm_entities)
        relation_result = self.merge_relations(
            natasha_relations or [],
            llm_relations or [],
            entity_id_map,
        )
        return EnsembleResult(
            entities=entity_result.entities,
            relations=relation_result.relations,
            entity_decisions=entity_result.entity_decisions,
            relation_decisions=relation_result.relation_decisions,
        )

    def merge_entities(
        self,
        natasha_entities: Iterable[EnrichedEntity],
        llm_entities: Iterable[EnrichedEntity],
    ) -> tuple[EnsembleResult, dict[str, str]]:
        grouped: dict[tuple[str, str], list[EnrichedEntity]] = defaultdict(list)
        originals: dict[str, str] = {}

        for ent in list(natasha_entities) + list(llm_entities):
            key = self._entity_key(ent)
            grouped[key].append(ent)

        merged_entities: list[EnrichedEntity] = []
        decisions: list[EnsembleDecision] = []
        entity_id_map: dict[str, str] = {}

        for index, (key, group) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
            merged = self._merge_entity_group(group, f"ens_e{index + 1}")
            merged_entities.append(merged)
            for ent in group:
                entity_id_map[ent.local_id] = merged.local_id
                originals[ent.local_id] = merged.local_id

            sources = tuple(sorted({self._source_name(ent.extractor) for ent in group}))
            single_source = len(sources) == 1
            type_conflict = len({str(ent.type) for ent in group}) > 1
            needs_review = (
                single_source and merged.confidence < self.single_source_review_threshold
            ) or type_conflict
            reason = (
                MergeReason.SINGLE_SOURCE
                if single_source
                else MergeReason.EXACT_CANONICAL_MATCH
            )
            notes: list[str] = []
            if type_conflict:
                notes.append("entity_type_conflict")

            decisions.append(EnsembleDecision(
                output_id=merged.local_id,
                sources=sources,
                confidence=merged.confidence,
                reason=reason,
                needs_review=needs_review,
                notes=tuple(notes),
            ))

        return EnsembleResult(
            entities=merged_entities,
            entity_decisions=decisions,
        ), entity_id_map

    def merge_relations(
        self,
        natasha_relations: Iterable[EnrichedRelation],
        llm_relations: Iterable[EnrichedRelation],
        entity_id_map: dict[str, str] | None = None,
    ) -> EnsembleResult:
        entity_id_map = entity_id_map or {}
        grouped: dict[tuple[str, str, str], list[EnrichedRelation]] = defaultdict(list)
        for rel in list(natasha_relations) + list(llm_relations):
            grouped[self._relation_key(rel, entity_id_map)].append(rel)

        merged_relations: list[EnrichedRelation] = []
        decisions: list[EnsembleDecision] = []

        for index, (key, group) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
            merged = self._merge_relation_group(group, f"ens_r{index + 1}", entity_id_map)
            merged_relations.append(merged)

            sources = tuple(sorted({self._source_name(rel.extractor) for rel in group}))
            single_source = len(sources) == 1
            needs_review = single_source and merged.confidence < self.single_source_review_threshold
            decisions.append(EnsembleDecision(
                output_id=f"{merged.source_local_id}->{merged.target_local_id}:{merged.relation_type}",
                sources=sources,
                confidence=merged.confidence,
                reason=(
                    MergeReason.SINGLE_SOURCE
                    if single_source
                    else MergeReason.RELATION_ENDPOINT_MATCH
                ),
                needs_review=needs_review,
            ))

        return EnsembleResult(
            relations=merged_relations,
            relation_decisions=decisions,
        )

    def _merge_entity_group(self, group: list[EnrichedEntity], local_id: str) -> EnrichedEntity:
        best = max(group, key=lambda ent: (ent.confidence, len(ent.entity)))
        confidence = self._aggregate_confidence(
            [(ent.confidence, self._extractor_weight(ent.extractor)) for ent in group]
        )
        mentions = self._unique(
            mention
            for ent in group
            for mention in ([ent.quote] + list(ent.mentions))
            if mention
        )
        char_start = min(ent.char_start for ent in group)
        char_end = max(ent.char_end for ent in group)
        extractors = ",".join(sorted({self._source_name(ent.extractor) for ent in group}))
        attributes = self._merge_attributes(group)

        return self._copy_entity(
            best,
            entity=self._select_canonical_name(group),
            confidence=confidence,
            local_id=local_id,
            mentions=mentions,
            attributes=attributes,
            char_start=char_start,
            char_end=char_end,
            extractor=f"ensemble:{extractors}",
            needs_review=len({str(ent.type) for ent in group}) > 1,
        )

    def _merge_relation_group(
        self,
        group: list[EnrichedRelation],
        relation_id: str,
        entity_id_map: dict[str, str],
    ) -> EnrichedRelation:
        best = max(group, key=lambda rel: (rel.confidence, len(rel.quote or "")))
        confidence = self._aggregate_confidence(
            [(rel.confidence, self._extractor_weight(rel.extractor)) for rel in group]
        )
        quote = self._select_quote([rel.quote for rel in group])
        extractors = ",".join(sorted({self._source_name(rel.extractor) for rel in group}))
        note = self._join_notes(rel.note for rel in group)

        return self._copy_relation(
            best,
            source_local_id=entity_id_map.get(best.source_local_id, best.source_local_id),
            target_local_id=entity_id_map.get(best.target_local_id, best.target_local_id),
            quote=quote,
            confidence=confidence,
            extractor=f"ensemble:{extractors}",
            needs_review=False,
            note=note,
        )

    def _entity_key(self, ent: EnrichedEntity) -> tuple[str, str]:
        return str(ent.type), canonicalize_text(ent.entity)

    def _relation_key(
        self,
        rel: EnrichedRelation,
        entity_id_map: dict[str, str],
    ) -> tuple[str, str, str]:
        source_id = entity_id_map.get(rel.source_local_id, rel.source_local_id)
        target_id = entity_id_map.get(rel.target_local_id, rel.target_local_id)
        if not source_id:
            source_id = f"{rel.source_entity_type}:{canonicalize_text(rel.source_entity)}"
        if not target_id:
            target_id = f"{rel.target_entity_type}:{canonicalize_text(rel.target_entity)}"
        return source_id, str(rel.relation_type), target_id

    def _aggregate_confidence(self, values: list[tuple[float, float]]) -> float:
        if not values:
            return 0.0
        weighted = sum(conf * weight for conf, weight in values)
        total_weight = sum(weight for _, weight in values)
        base = weighted / total_weight if total_weight else 0.0
        multi_source_bonus = 0.08 if len(values) > 1 else 0.0
        return min(1.0, round(base + multi_source_bonus, 4))

    def _extractor_weight(self, extractor: str) -> float:
        normalized = self._source_name(extractor)
        if normalized == EvidenceSource.LLM:
            return self.llm_weight
        if normalized == EvidenceSource.NATASHA:
            return self.natasha_weight
        return max(self.llm_weight, self.natasha_weight)

    def _source_name(self, extractor: str) -> str:
        text = (extractor or "").lower()
        if "natasha" in text:
            return EvidenceSource.NATASHA
        if "llm" in text or "yandex" in text:
            return EvidenceSource.LLM
        if "ensemble" in text:
            return EvidenceSource.ENSEMBLE
        return extractor or "unknown"

    def _select_canonical_name(self, group: list[EnrichedEntity]) -> str:
        candidates = sorted(
            group,
            key=lambda ent: (ent.confidence, len(canonicalize_text(ent.entity))),
            reverse=True,
        )
        return candidates[0].entity

    def _select_quote(self, quotes: Iterable[str | None]) -> str:
        clean = [quote.strip() for quote in quotes if quote and quote.strip()]
        if not clean:
            return ""
        return max(clean, key=len)

    def _merge_attributes(self, group: list[EnrichedEntity]) -> dict:
        merged: dict = {}
        for ent in sorted(group, key=lambda item: item.confidence):
            merged.update(ent.attributes or {})
        return merged

    def _join_notes(self, notes: Iterable[str | None]) -> str | None:
        clean = self._unique(note.strip() for note in notes if note and note.strip())
        return "; ".join(clean) if clean else None

    def _unique(self, values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out

    def _copy_entity(self, source: EnrichedEntity, **updates) -> EnrichedEntity:
        try:
            return source.model_copy(update=updates)
        except AttributeError:
            data = source.model_dump()
            data.update(updates)
            return EnrichedEntity(**data)

    def _copy_relation(self, relation: EnrichedRelation, **updates) -> EnrichedRelation:
        try:
            return relation.model_copy(update=updates)
        except AttributeError:
            data = relation.model_dump()
            data.update(updates)
            return EnrichedRelation(**data)
