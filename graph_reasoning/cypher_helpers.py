"""Cypher query helpers for the Nornikel Knowledge Graph.

Each function returns a list of result rows. Async, expects an
async-compatible Neo4j driver (``neo4j.AsyncDriver``).

All queries respect the new per-type label schema
(``(:Material)``, ``(:Process)``, …) introduced in the loader
refactor.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ontology import EntityType, label_for

log = logging.getLogger(__name__)


# -------------------------------------------------------------- by seed name


async def entities_by_name(
    driver: Any,
    names: Sequence[str],
    *,
    entity_type: EntityType | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find nodes by ``name`` (case-insensitive exact match).

    If ``entity_type`` is given, the query is restricted to that
    label; otherwise any label is searched.
    """
    if not names:
        return []
    label = f":{label_for(entity_type).value}" if entity_type is not None else ""
    cypher = f"""
    MATCH (n{label})
    WHERE toLower(n.name) IN $names_lc
    RETURN n.name AS name, labels(n) AS labels, n.confidence AS confidence
    LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(
            cypher, names_lc=[n.lower() for n in names], limit=limit
        )
        return [dict(record) async for record in result]


# -------------------------------------------------------------- by geography


async def entities_by_geography(
    driver: Any,
    geography_name: str,
    *,
    entity_type: EntityType | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find entities linked to a given Geography node."""
    if not geography_name:
        return []
    label = f":{label_for(entity_type).value}" if entity_type is not None else ""
    cypher = f"""
    MATCH (n{label})-[:HAS_GEOGRAPHY]->(g:Geography)
    WHERE toLower(g.name) = toLower($name)
    RETURN n.name AS name, labels(n) AS labels,
           g.name AS geography, g.kind AS geo_kind
    LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(
            cypher, name=geography_name, limit=limit
        )
        return [dict(record) async for record in result]


# -------------------------------------------------------------- by year range


async def entities_by_year_range(
    driver: Any,
    *,
    min_year: int | None = None,
    max_year: int | None = None,
    entity_type: EntityType | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find entities whose PUBLISHED_IN_YEAR falls in [min_year, max_year]."""
    if min_year is None and max_year is None:
        return []
    label = f":{label_for(entity_type).value}" if entity_type is not None else ""
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit}
    if min_year is not None:
        conditions.append("y.value >= $min_year")
        params["min_year"] = min_year
    if max_year is not None:
        conditions.append("y.value <= $max_year")
        params["max_year"] = max_year
    where = " AND ".join(conditions)
    cypher = f"""
    MATCH (n{label})-[:PUBLISHED_IN_YEAR]->(y:Year)
    WHERE {where}
    RETURN n.name AS name, labels(n) AS labels, y.value AS year
    LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(cypher, **params)
        return [dict(record) async for record in result]


# -------------------------------------------------------------- by numeric value


async def entities_by_numeric_value(
    driver: Any,
    *,
    property_name: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    unit: str | None = None,
    entity_type: EntityType | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find entities linked to a NumericValue matching the given range."""
    label = f":{label_for(entity_type).value}" if entity_type is not None else ""
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit}
    if min_value is not None:
        conditions.append("n.numeric_value >= $min_value")
        params["min_value"] = float(min_value)
    if max_value is not None:
        conditions.append("n.numeric_value <= $max_value")
        params["max_value"] = float(max_value)
    if property_name:
        conditions.append("toLower(n.property_name) = toLower($property_name)")
        params["property_name"] = property_name
    if unit:
        conditions.append("toLower(n.unit) = toLower($unit)")
        params["unit"] = unit
    where = " AND ".join(conditions) if conditions else "TRUE"
    cypher = f"""
    MATCH (e{label})-[:HAS_NUMERIC_VALUE]->(n:NumericValue)
    WHERE {where}
    RETURN e.name AS entity_name, labels(e) AS labels,
           n.numeric_value AS value, n.unit AS unit,
           n.min_value AS range_min, n.max_value AS range_max,
           n.operator AS operator, n.property_name AS property_name
    LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(cypher, **params)
        return [dict(record) async for record in result]


# -------------------------------------------------------------- top relations


async def top_related(
    driver: Any,
    seed_name: str,
    *,
    relation_type: str | None = None,
    max_hops: int = 2,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find entities most related to ``seed_name`` within ``max_hops`` hops."""
    if not seed_name:
        return []
    rel_clause = f":{relation_type.upper()}" if relation_type else ""
    cypher = f"""
    MATCH (seed)-[r{rel_clause}*1..{max_hops}]-(related)
    WHERE toLower(seed.name) = toLower($seed)
      AND related <> seed
    RETURN related.name AS name, labels(related) AS labels,
           count(DISTINCT r) AS path_count
    ORDER BY path_count DESC
    LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(cypher, seed=seed_name, limit=limit)
        return [dict(record) async for record in result]


# -------------------------------------------------------------- graph statistics


async def graph_statistics(driver: Any) -> dict[str, int]:
    """Return a dict ``{label: count}`` plus total node / edge counts."""
    from ontology import EntityLabel

    out: dict[str, int] = {"total_nodes": 0, "total_relationships": 0}
    async with driver.session() as session:
        for label in EntityLabel:
            result = await session.run(
                f"MATCH (n:{label.value}) RETURN count(n) AS c"
            )
            record = await result.single()
            out[label.value] = int(record["c"]) if record else 0
        rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) AS c")
        rel_record = await rel_result.single()
        out["total_relationships"] = int(rel_record["c"]) if rel_record else 0
        # Total node count.
        node_result = await session.run("MATCH (n) RETURN count(n) AS c")
        node_record = await node_result.single()
        out["total_nodes"] = int(node_record["c"]) if node_record else 0
    return out


__all__ = [
    "entities_by_geography",
    "entities_by_name",
    "entities_by_numeric_value",
    "entities_by_year_range",
    "graph_statistics",
    "top_related",
]
