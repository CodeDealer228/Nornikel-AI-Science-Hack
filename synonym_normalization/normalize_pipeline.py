from typing import List

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation

from .entity_matcher import deduplicate_entities
from .synonym_dictionary import SynonymDictionary
from .units_normalization import normalize_unit


def normalize_entities(
    entities: List[EnrichedEntity],
    syn_dict: SynonymDictionary,
) -> List[EnrichedEntity]:
    """
    Resolve aliases, normalize units in Property attributes, and deduplicate
    entities within the current chunk.
    """
    for ent in entities:
        ent.entity = syn_dict.resolve(ent.entity)
        syn_dict.add_term(ent.entity, ent.mentions)

        if str(ent.type) == "Property" and "unit" in ent.attributes:
            ent.attributes["unit"] = normalize_unit(ent.attributes.get("unit"))

    return deduplicate_entities(entities)


def normalize_relations(
    relations: List[EnrichedRelation],
    syn_dict: SynonymDictionary,
) -> List[EnrichedRelation]:
    """Resolve entity names used on relation endpoints."""
    for rel in relations:
        rel.source_entity = syn_dict.resolve(rel.source_entity)
        rel.target_entity = syn_dict.resolve(rel.target_entity)

    return relations
