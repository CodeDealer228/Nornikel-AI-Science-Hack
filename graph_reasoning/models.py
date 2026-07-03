from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphNode:
    id: str
    name: str
    type: str
    source_documents: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation_type: str
    quote: str = ""
    confidence: float = 0.0
    source_document: str | None = None
    chunk_id: str | None = None


@dataclass(frozen=True)
class GraphPath:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


@dataclass(frozen=True)
class GraphGap:
    code: str
    message: str
    entity_id: str | None = None
    severity: str = "warning"


@dataclass
class GraphReasoningContext:
    seed_entities: tuple[str, ...]
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    contradictions: list[GraphEdge] = field(default_factory=list)
    gaps: list[GraphGap] = field(default_factory=list)

    def node_by_id(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}
