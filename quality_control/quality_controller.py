from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation

from .models import QualityIssue, QualityReport


class FactQualityController:
    def __init__(self, low_confidence_threshold: float = 0.45) -> None:
        self.low_confidence_threshold = low_confidence_threshold

    def inspect(
        self,
        entities: Iterable[EnrichedEntity],
        relations: Iterable[EnrichedRelation],
    ) -> QualityReport:
        entity_list = list(entities)
        relation_list = list(relations)
        issues: list[QualityIssue] = []
        issues.extend(self.detect_low_confidence(entity_list, relation_list))
        issues.extend(self.detect_relation_endpoint_gaps(entity_list, relation_list))
        issues.extend(self.detect_extracted_contradictions(relation_list))
        issues.extend(self.detect_sparse_graph_signals(entity_list, relation_list))
        return QualityReport(issues=issues)

    def calibrate_entity_confidence(self, entity: EnrichedEntity, support_count: int = 1) -> float:
        source_bonus = 0.08 if "ensemble" in (entity.extractor or "").lower() else 0.0
        support_bonus = min(0.12, max(0, support_count - 1) * 0.03)
        review_penalty = 0.12 if entity.needs_review else 0.0
        return min(1.0, max(0.0, round(entity.confidence + source_bonus + support_bonus - review_penalty, 4)))

    def calibrate_relation_confidence(self, relation: EnrichedRelation, support_count: int = 1) -> float:
        quote_bonus = 0.05 if relation.quote else 0.0
        source_bonus = 0.08 if "ensemble" in (relation.extractor or "").lower() else 0.0
        support_bonus = min(0.12, max(0, support_count - 1) * 0.03)
        review_penalty = 0.12 if relation.needs_review else 0.0
        return min(1.0, max(0.0, round(
            relation.confidence + quote_bonus + source_bonus + support_bonus - review_penalty,
            4,
        )))

    def detect_low_confidence(
        self,
        entities: list[EnrichedEntity],
        relations: list[EnrichedRelation],
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for entity in entities:
            if entity.confidence < self.low_confidence_threshold:
                issues.append(QualityIssue(
                    code="low_confidence_entity",
                    severity="warning",
                    message=f"Entity '{entity.entity}' confidence is {entity.confidence:.2f}.",
                    evidence_ids=(entity.local_id,),
                ))
        for index, relation in enumerate(relations):
            if relation.confidence < self.low_confidence_threshold:
                issues.append(QualityIssue(
                    code="low_confidence_relation",
                    severity="warning",
                    message=(
                        f"Relation '{relation.source_entity} {relation.relation_type} "
                        f"{relation.target_entity}' confidence is {relation.confidence:.2f}."
                    ),
                    evidence_ids=(str(index),),
                ))
        return issues

    def detect_relation_endpoint_gaps(
        self,
        entities: list[EnrichedEntity],
        relations: list[EnrichedRelation],
    ) -> list[QualityIssue]:
        ids = {entity.local_id for entity in entities}
        issues: list[QualityIssue] = []
        for index, relation in enumerate(relations):
            missing = []
            if relation.source_local_id not in ids:
                missing.append(relation.source_local_id)
            if relation.target_local_id not in ids:
                missing.append(relation.target_local_id)
            if missing:
                issues.append(QualityIssue(
                    code="missing_relation_endpoint",
                    severity="error",
                    message=f"Relation endpoint ids are missing from entity set: {missing}",
                    evidence_ids=(str(index),),
                ))
        return issues

    def detect_extracted_contradictions(self, relations: list[EnrichedRelation]) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        for index, relation in enumerate(relations):
            if str(relation.relation_type) == "contradicts":
                issues.append(QualityIssue(
                    code="explicit_contradiction",
                    severity="warning",
                    message=(
                        f"Explicit contradiction: {relation.source_entity} contradicts "
                        f"{relation.target_entity}."
                    ),
                    evidence_ids=(str(index),),
                ))

        by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
        for relation in relations:
            pair = tuple(sorted((relation.source_local_id, relation.target_local_id)))
            by_pair[pair].add(str(relation.relation_type))

        for pair, predicates in by_pair.items():
            if {"has_limitation", "has_expected_result"} <= predicates:
                issues.append(QualityIssue(
                    code="possible_semantic_contradiction",
                    severity="warning",
                    message="Same entity pair has both limitation and expected-result facts.",
                    evidence_ids=pair,
                ))
        return issues

    def detect_sparse_graph_signals(
        self,
        entities: list[EnrichedEntity],
        relations: list[EnrichedRelation],
    ) -> list[QualityIssue]:
        if not entities:
            return [QualityIssue(
                code="no_entities",
                severity="warning",
                message="No entities were extracted from the chunk.",
            )]
        if entities and not relations:
            return [QualityIssue(
                code="entities_without_relations",
                severity="warning",
                message="Entities were extracted, but no relations connect them.",
            )]
        return []
