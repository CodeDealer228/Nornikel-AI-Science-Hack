from __future__ import annotations

from typing import Any, Dict, List, Literal

try:
    from pydantic import BaseModel, Field
except Exception:
    class _Field:
        def __init__(self, default=None, default_factory=None):
            self.default = default
            self.default_factory = default_factory

    def Field(default=None, default_factory=None):
        return _Field(default=default, default_factory=default_factory)

    class BaseModel:
        def __init__(self, **data):
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
        def _field_names(cls):
            names = []
            for base in reversed(cls.__mro__):
                names.extend(getattr(base, "__annotations__", {}).keys())
            return names

        def model_dump(self):
            return {name: _dump(getattr(self, name)) for name in self._field_names()}

    def _dump(value):
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [_dump(v) for v in value]
        if isinstance(value, dict):
            return {k: _dump(v) for k, v in value.items()}
        return value

class PrimaryEntity(BaseModel):
    text: str
    normal: str
    type: Literal["PER", "LOC", "ORG"]
    start: int
    stop: int

class NatashaAnnotation(BaseModel):
    n_sentences: int
    n_tokens: int
    ner_available: bool = True
    primary_entities: List[PrimaryEntity] = Field(default_factory=list)
    lemmas: List[str] = Field(default_factory=list)

class ChunkProvenance(BaseModel):
    source_document: str
    char_start: int
    char_end: int
    heading_path: List[str] = Field(default_factory=list)

class Chunk(BaseModel):
    chunk_id: str
    index: int
    provenance: ChunkProvenance
    text: str
    overlap_prefix_chars: int = 0
    oversize: bool = False
    natasha: NatashaAnnotation
    doc_metadata: Dict[str, Any] = Field(default_factory=dict)
