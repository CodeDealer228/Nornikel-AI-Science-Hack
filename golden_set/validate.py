from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .schema import ENTITY_TYPES, RELATION_TYPES, REQUIRED_SAMPLE_FIELDS


def load_jsonl(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number}: JSONL item must be an object")
            samples.append(item)
    return samples


def validate_sample(sample: dict, line_number: int) -> list[str]:
    sample_id = sample.get("sample_id", "<missing sample_id>")
    prefix = f"line {line_number}, sample_id={sample_id!r}"
    errors: list[str] = []

    missing = sorted(REQUIRED_SAMPLE_FIELDS - sample.keys())
    for field in missing:
        errors.append(f"{prefix}: missing required sample field {field!r}")

    for field in ("sample_id", "chunk_id", "document_id", "source_path", "text"):
        if field in sample and not _non_empty_str(sample[field]):
            errors.append(f"{prefix}: sample field {field!r} must be a non-empty string")

    text = sample.get("text")
    if not isinstance(text, str):
        text = ""

    entities = sample.get("entities")
    if not isinstance(entities, list):
        errors.append(f"{prefix}: 'entities' must be a list")
        entities = []

    relations = sample.get("relations")
    if not isinstance(relations, list):
        errors.append(f"{prefix}: 'relations' must be a list")
        relations = []

    entity_ids: set[str] = set()
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            errors.append(f"{prefix}: entity[{index}] must be an object")
            continue
        errors.extend(_validate_entity(entity, index, text, entity_ids, prefix))

    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            errors.append(f"{prefix}: relation[{index}] must be an object")
            continue
        errors.extend(_validate_relation(relation, index, text, entity_ids, prefix))

    return errors


def validate_file(path: Path) -> int:
    errors: list[str] = []
    try:
        samples = load_jsonl(path)
    except OSError as exc:
        print(f"{path}: cannot read file: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for line_number, sample in enumerate(samples, start=1):
        errors.extend(validate_sample(sample, line_number))

    if errors:
        for error in errors:
            print(error)
        return 1

    print("Golden set is valid")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate golden set JSONL")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    raise SystemExit(validate_file(args.path))


def _validate_entity(
    entity: dict[str, Any],
    index: int,
    sample_text: str,
    entity_ids: set[str],
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    required = ("id", "type", "canonical_name")
    for field in required:
        if field not in entity:
            errors.append(f"{prefix}: entity[{index}] missing required field {field!r}")

    entity_id = entity.get("id")
    if not _non_empty_str(entity_id):
        errors.append(f"{prefix}: entity[{index}].id must be a non-empty string")
    elif entity_id in entity_ids:
        errors.append(f"{prefix}: duplicate entity id {entity_id!r}")
    else:
        entity_ids.add(entity_id)

    entity_type = entity.get("type")
    if entity_type not in ENTITY_TYPES:
        errors.append(f"{prefix}: entity[{index}].type {entity_type!r} is not allowed")

    if not _non_empty_str(entity.get("canonical_name")):
        errors.append(f"{prefix}: entity[{index}].canonical_name must be a non-empty string")

    mentions = entity.get("mentions")
    if mentions is not None:
        if not isinstance(mentions, list):
            errors.append(f"{prefix}: entity[{index}].mentions must be a list")
        else:
            for mention_index, mention in enumerate(mentions):
                errors.extend(
                    _validate_mention(mention, index, mention_index, sample_text, prefix)
                )

    attributes = entity.get("attributes")
    if attributes is not None and not isinstance(attributes, dict):
        errors.append(f"{prefix}: entity[{index}].attributes must be an object")

    return errors


def _validate_mention(
    mention: Any,
    entity_index: int,
    mention_index: int,
    sample_text: str,
    prefix: str,
) -> list[str]:
    path = f"entity[{entity_index}].mentions[{mention_index}]"
    errors: list[str] = []
    if not isinstance(mention, dict):
        return [f"{prefix}: {path} must be an object"]

    mention_text = mention.get("text")
    if not _non_empty_str(mention_text):
        errors.append(f"{prefix}: {path}.text must be a non-empty string")
        return errors

    if mention_text not in sample_text:
        errors.append(f"{prefix}: {path}.text is not an exact substring of sample text")

    if "start" in mention or "end" in mention:
        start = mention.get("start")
        end = mention.get("end")
        if not _valid_span(start, end, sample_text):
            errors.append(f"{prefix}: {path}.start/end must be valid integer offsets")
        elif sample_text[start:end] != mention_text:
            errors.append(f"{prefix}: {path}.start/end slice does not match mention.text")

    return errors


def _validate_relation(
    relation: dict[str, Any],
    index: int,
    sample_text: str,
    entity_ids: set[str],
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    required = ("id", "subject", "predicate", "object")
    for field in required:
        if field not in relation:
            errors.append(f"{prefix}: relation[{index}] missing required field {field!r}")

    relation_id = relation.get("id")
    if not _non_empty_str(relation_id):
        errors.append(f"{prefix}: relation[{index}].id must be a non-empty string")

    predicate = relation.get("predicate")
    if predicate not in RELATION_TYPES:
        errors.append(f"{prefix}: relation[{index}].predicate {predicate!r} is not allowed")

    subject = relation.get("subject")
    if subject not in entity_ids:
        errors.append(f"{prefix}: relation[{index}].subject {subject!r} is unknown")

    obj = relation.get("object")
    if obj not in entity_ids:
        errors.append(f"{prefix}: relation[{index}].object {obj!r} is unknown")

    evidence_text = relation.get("evidence_text")
    if evidence_text is not None:
        if not _non_empty_str(evidence_text):
            errors.append(f"{prefix}: relation[{index}].evidence_text must be a non-empty string")
        elif evidence_text not in sample_text:
            errors.append(
                f"{prefix}: relation[{index}].evidence_text is not an exact substring of sample text"
            )

    if "evidence_start" in relation or "evidence_end" in relation:
        start = relation.get("evidence_start")
        end = relation.get("evidence_end")
        if evidence_text is None:
            errors.append(
                f"{prefix}: relation[{index}] has evidence offsets but no evidence_text"
            )
        elif not _valid_span(start, end, sample_text):
            errors.append(
                f"{prefix}: relation[{index}].evidence_start/evidence_end must be valid integer offsets"
            )
        elif sample_text[start:end] != evidence_text:
            errors.append(
                f"{prefix}: relation[{index}].evidence_start/evidence_end slice does not match evidence_text"
            )

    return errors


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_span(start: Any, end: Any, text: str) -> bool:
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start <= end <= len(text)
    )


if __name__ == "__main__":
    main()
