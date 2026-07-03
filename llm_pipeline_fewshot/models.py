from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except Exception:
    class _Field:
        def __init__(self, default=None, default_factory=None, **_: Any) -> None:
            self.default = default
            self.default_factory = default_factory

    def Field(default=None, default_factory=None, **kwargs: Any):
        return _Field(default=default, default_factory=default_factory, **kwargs)

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    def field_validator(*_: Any, **__: Any):
        def decorator(fn):
            return fn
        return decorator

    def model_validator(*_: Any, **__: Any):
        def decorator(fn):
            return fn
        return decorator

    class BaseModel:
        def __init__(self, **data: Any) -> None:
            for name in self._field_names():
                default = getattr(self.__class__, name, None)
                if name in data:
                    value = data[name]
                elif isinstance(default, _Field) and default.default_factory is not None:
                    value = default.default_factory()
                elif isinstance(default, _Field):
                    value = default.default
                elif default is not None:
                    value = default
                else:
                    value = None
                setattr(self, name, value)

        @classmethod
        def _field_names(cls) -> list[str]:
            names: list[str] = []
            for base in reversed(cls.__mro__):
                names.extend(getattr(base, "__annotations__", {}).keys())
            return names

        def model_dump(self) -> dict[str, Any]:
            return {name: _dump(getattr(self, name)) for name in self._field_names()}

    def _dump(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [_dump(v) for v in value]
        if isinstance(value, dict):
            return {k: _dump(v) for k, v in value.items()}
        return value


class EntityType(StrEnum):
    MATERIAL = "Material"
    SUBSTANCE = "Substance"
    PROCESS = "Process"
    EQUIPMENT = "Equipment"
    PROPERTY = "Property"
    PARAMETER = "Parameter"
    CONDITION = "Condition"
    EXPERIMENT = "Experiment"
    PUBLICATION = "Publication"
    TECHNOLOGY_SOLUTION = "TechnologySolution"
    RESULT = "Result"
    CONCLUSION = "Conclusion"
    LIMITATION = "Limitation"
    FACILITY = "Facility"
    ORGANIZATION = "Organization"
    EXPERT = "Expert"


class RelationType(StrEnum):
    HAS_SUBPROCESS = "has_subprocess"
    REPLACED_BY = "replaced_by"
    AFFECTS_PROPERTY = "affects_property"
    HAS_LIMITATION = "has_limitation"
    HAS_QUALITY_REQUIREMENT = "has_quality_requirement"
    HAS_DISTRIBUTION_REQUIREMENT = "has_distribution_requirement"
    HAS_EXPECTED_RESULT = "has_expected_result"
    BASED_ON = "based_on"
    APPLIES_TO = "applies_to"
    USES_TECHNOLOGY = "uses_technology"
    PRODUCES_OUTPUT = "produces_output"
    REQUIRES_EXPERTISE = "requires_expertise"
    USES_EQUIPMENT = "uses_equipment"
    STUDIES = "studies"
    FED_THROUGH = "fed_through"
    DEPENDS_ON = "depends_on"
    OPERATES_AT_CONDITION = "operates_at_condition"
    MEASURED_PROPERTY = "measured_property"
    OPERATES_BETWEEN = "operates_between"
    VALIDATED_BY = "validated_by"
    PERFORMED_BY = "performed_by"
    USES_MATERIAL = "uses_material"
    DESCRIBED_IN = "described_in"
    SUPPORTED_BY = "supported_by"
    AUTHORED_BY = "authored_by"
    AFFILIATED_WITH = "affiliated_with"
    USED_IN_FACILITY = "used_in_facility"
    CONTRADICTS = "contradicts"


class BaseStrictModel(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class ChunkProvenance(BaseStrictModel):
    source_document: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    pages: list[int] | None = None

    @model_validator(mode="after")
    def validate_span(self) -> "ChunkProvenance":
        if self.char_end < self.char_start:
            raise ValueError("char_end must be greater than or equal to char_start")
        return self


class ChunkInput(BaseStrictModel):
    chunk_id: str = Field(min_length=1)
    index: int = 0
    provenance: ChunkProvenance
    text: str = Field(min_length=1)
    overlap_prefix_chars: int = Field(default=0, ge=0)
    oversize: bool = False
    natasha: dict[str, Any] = Field(default_factory=dict)
    doc_metadata: dict[str, Any] = Field(default_factory=dict)


class LLMEntity(BaseStrictModel):
    local_id: str = Field(min_length=1)
    type: EntityType
    canonical_name: str = Field(min_length=1)
    mentions: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("local_id", "canonical_name", mode="before")
    @classmethod
    def strip_required_string(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("mentions", mode="before")
    @classmethod
    def coerce_mentions(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value) if isinstance(value, (tuple, set)) else value

    @field_validator("mentions")
    @classmethod
    def normalize_mentions(cls, value: list[Any]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for mention in value:
            mention_text = str(mention).strip()
            if mention_text and mention_text not in seen:
                seen.add(mention_text)
                normalized.append(mention_text)
        return normalized


class LLMRelation(BaseStrictModel):
    subject: str = Field(min_length=1)
    predicate: RelationType
    object: str = Field(min_length=1)
    quote: str | None = None
    note: str | None = None

    @field_validator("subject", "object", "quote", "note", mode="before")
    @classmethod
    def strip_optional_string(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class LLMExtraction(BaseStrictModel):
    entities: list[LLMEntity] = Field(default_factory=list)
    relations: list[LLMRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_entity_ids(self) -> "LLMExtraction":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for entity in self.entities:
            if entity.local_id in seen:
                duplicates.add(entity.local_id)
            seen.add(entity.local_id)
        if duplicates:
            raise ValueError(f"duplicate entity local_id values: {sorted(duplicates)}")
        return self


class ValidationIssue(BaseStrictModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    local_id: str | None = None
    relation_index: int | None = None


class Provenance(BaseStrictModel):
    chunk_id: str
    source_document: str
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    char_start: int
    char_end: int
    quote: str


class EnrichedEntity(BaseStrictModel):
    entity: str
    type: EntityType
    chunk_id: str
    source_document: str
    page: int | None = None
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)
    local_id: str
    mentions: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    heading_path: list[str] = Field(default_factory=list)
    char_start: int
    char_end: int
    extractor: str = "yandex_llm"
    needs_review: bool = False
    provenance: Provenance | None = None


class EnrichedRelation(BaseStrictModel):
    source_entity: str
    target_entity: str
    relation_type: RelationType
    chunk_id: str
    source_document: str
    page: int | None = None
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_entity_type: EntityType
    target_entity_type: EntityType
    source_local_id: str
    target_local_id: str
    heading_path: list[str] = Field(default_factory=list)
    char_start: int
    char_end: int
    extractor: str = "yandex_llm"
    needs_review: bool = False
    note: str | None = None
    provenance: Provenance | None = None


class ChunkExtractionResult(BaseStrictModel):
    chunk_id: str
    status: Literal["ok", "empty", "error"]
    source_document: str
    char_start: int
    char_end: int
    page: int | None = None
    model_uri: str | None = None
    provider: str = "yandex_foundation_models"
    attempts: int = 0
    entities: list[EnrichedEntity] = Field(default_factory=list)
    relations: list[EnrichedRelation] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    raw_response: str | None = None
    error: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
