from __future__ import annotations

import re
from typing import Any, Iterable

from .models import GraphEdge, GraphNode, GraphPath, GraphReasoningContext


_GENERIC_EXACT_SEEDS = frozenset({"вода"})


class Neo4jSubgraphExtractor:
    """
    Async Neo4j subgraph reader. It expects an object compatible with
    neo4j.AsyncDriver but does not import or instantiate the driver itself.
    """

    def __init__(self, driver: Any) -> None:
        self.driver = driver

    async def extract_subgraph(
        self,
        seed_entity_names: Iterable[str],
        max_hops: int = 3,
        limit: int = 200,
    ) -> GraphReasoningContext:
        hops = min(max(max_hops, 1), 4)
        seeds = tuple(seed_entity_names)
        seed_names_lc = [
            seed.lower()
            for seed in seeds
            if seed and seed.lower() not in _GENERIC_EXACT_SEEDS
        ]
        seed_terms_lc = self._seed_terms(seeds)

        exact_query = f"""
        MATCH (seed)
        WHERE seed.type IS NOT NULL
          AND toLower(seed.name) IN $seed_names_lc
        MATCH path = (seed)-[*1..{hops}]-(neighbor)
        WHERE all(node IN nodes(path) WHERE node.type IS NOT NULL)
        WITH path LIMIT $limit
        RETURN path
        """

        fuzzy_query = f"""
        MATCH (seed)
        WHERE seed.type IS NOT NULL
        WITH seed, toLower(seed.name) AS seed_name
        WITH seed,
             reduce(score = 0, term IN $seed_terms_lc |
                 score + CASE WHEN seed_name CONTAINS term THEN size(term) ELSE 0 END
             ) AS seed_score
        WHERE seed_score > 0
        ORDER BY seed_score DESC, size(seed.name) ASC
        WITH seed LIMIT $seed_limit
        MATCH path = (seed)-[*1..{hops}]-(neighbor)
        WHERE all(node IN nodes(path) WHERE node.type IS NOT NULL)
        WITH path LIMIT $limit
        RETURN path
        """

        context = await self._run_path_query(
            exact_query,
            seeds=seeds,
            seed_names_lc=seed_names_lc,
            seed_terms_lc=seed_terms_lc,
            limit=limit,
        )
        if context.paths or not seed_terms_lc:
            return context

        return await self._run_path_query(
            fuzzy_query,
            seeds=seeds,
            seed_names_lc=seed_names_lc,
            seed_terms_lc=seed_terms_lc,
            limit=limit,
            seed_limit=25,
        )

    async def _run_path_query(
        self,
        query: str,
        *,
        seeds: tuple[str, ...],
        seed_names_lc: list[str],
        seed_terms_lc: list[str],
        limit: int,
        seed_limit: int = 100,
    ) -> GraphReasoningContext:
        nodes: dict[str, GraphNode] = {}
        edges: dict[tuple[str, str, str, str], GraphEdge] = {}
        paths: list[GraphPath] = []

        async with self.driver.session() as session:
            result = await session.run(
                query,
                seed_names_lc=seed_names_lc,
                seed_terms_lc=seed_terms_lc,
                limit=limit,
                seed_limit=seed_limit,
            )
            async for record in result:
                path = record["path"]
                path_nodes = [self._node_from_neo4j(node) for node in path.nodes]
                path_edges = [self._edge_from_neo4j(rel) for rel in path.relationships]
                for node in path_nodes:
                    nodes[node.id] = node
                for edge in path_edges:
                    edges[(edge.source_id, edge.target_id, edge.relation_type, edge.quote)] = edge
                paths.append(GraphPath(tuple(path_nodes), tuple(path_edges)))

        return GraphReasoningContext(
            seed_entities=seeds,
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            paths=paths,
        )

    @staticmethod
    def _seed_terms(seeds: Iterable[str]) -> list[str]:
        terms: list[str] = []
        for seed in seeds:
            normalized = str(seed or "").lower()
            for token in re.findall(r"[a-zа-яё0-9]{3,}", normalized, flags=re.IGNORECASE):
                candidates = {token}
                if len(token) >= 6:
                    candidates.add(token[:5])
                if len(token) == 4:
                    candidates.add(token[:3])
                for candidate in candidates:
                    if candidate and candidate not in terms:
                        terms.append(candidate)
        return terms

    def _node_from_neo4j(self, node: Any) -> GraphNode:
        props = dict(node)
        element_id = getattr(node, "element_id", None) or str(props.get("id") or props.get("name"))
        return GraphNode(
            id=element_id,
            name=str(props.get("name") or props.get("id") or element_id),
            type=str(props.get("type") or "Entity"),
            source_documents=tuple(props.get("source_documents") or ()),
            confidence=float(props.get("confidence") or 0.0),
        )

    def _edge_from_neo4j(self, rel: Any) -> GraphEdge:
        props = dict(rel)
        start_id = getattr(rel, "start_node", None)
        end_id = getattr(rel, "end_node", None)
        source_id = getattr(start_id, "element_id", None) or str(props.get("source_id") or "")
        target_id = getattr(end_id, "element_id", None) or str(props.get("target_id") or "")
        return GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=str(getattr(rel, "type", None) or props.get("type") or ""),
            quote=str(props.get("quote") or ""),
            confidence=float(props.get("confidence") or 0.0),
            source_document=props.get("source_document"),
            chunk_id=props.get("chunk_id"),
        )
