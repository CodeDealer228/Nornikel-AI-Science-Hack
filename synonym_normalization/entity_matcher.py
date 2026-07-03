from typing import List

from llm_pipeline_fewshot.models import EnrichedEntity

from .canonicalizer import canonicalize_text


def deduplicate_entities(entities: List[EnrichedEntity]) -> List[EnrichedEntity]:
    """
    Group entities with the same canonical text and type within a chunk,
    merging mentions and widening provenance offsets.
    """
    merged = {}
    for ent in entities:
        key = (ent.type, canonicalize_text(ent.entity))
        if key not in merged:
            merged[key] = ent
            continue

        existing = merged[key]
        existing.mentions = list(dict.fromkeys(existing.mentions + ent.mentions))
        existing.char_start = min(existing.char_start, ent.char_start)
        existing.char_end = max(existing.char_end, ent.char_end)

    return list(merged.values())
