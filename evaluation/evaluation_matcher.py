from typing import List, Tuple

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation
from synonym_normalization.canonicalizer import canonicalize_text


def _hash_entity(ent: EnrichedEntity) -> str:
    return f"{ent.type}:{canonicalize_text(ent.entity)}"


def _hash_relation(rel: EnrichedRelation) -> str:
    src = f"{rel.source_entity_type}:{canonicalize_text(rel.source_entity)}"
    tgt = f"{rel.target_entity_type}:{canonicalize_text(rel.target_entity)}"
    return f"{src}-[{rel.relation_type}]->{tgt}"


def compare_entities(
    predicted: List[EnrichedEntity],
    expected: List[EnrichedEntity],
) -> Tuple[int, int, int]:
    pred_set = {_hash_entity(e) for e in predicted}
    exp_set = {_hash_entity(e) for e in expected}

    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    return tp, fp, fn


def compare_relations(
    predicted: List[EnrichedRelation],
    expected: List[EnrichedRelation],
) -> Tuple[int, int, int]:
    pred_set = {_hash_relation(r) for r in predicted}
    exp_set = {_hash_relation(r) for r in expected}

    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    return tp, fp, fn
