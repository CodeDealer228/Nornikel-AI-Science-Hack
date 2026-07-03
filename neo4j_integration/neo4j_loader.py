"""Neo4j loader with proper per-type labels and first-class geography/year/numeric nodes.

Schema overview
---------------
Each ``EntityType`` (Material, Process, …) maps to its own Neo4j
label. This means a query like ``MATCH (m:Material) RETURN m``
is fast and type-safe (no property filter needed).

Auxiliary nodes:
    * ``:Document`` — one per source document
    * ``:Chunk`` — one per text chunk
    * ``:Alias`` — known alternative names (synonym dictionary)
    * ``:Geography`` — geographic anchor (Russia, Worldwide, …)
    * ``:Year`` — temporal anchor (publication year)
    * ``:NumericValue`` — numeric measurement (with min/max/unit/operator)

Relationships:
    * ``(doc:Document)-[:HAS_CHUNK]->(c:Chunk)``
    * ``(c:Chunk)-[:MENTIONS]->(e:Entity)``
    * ``(e:Entity)-[:KNOWN_AS]->(a:Alias)``
    * ``(c:Chunk)-[:SUPPORTS]->(r:REL_TYPE)``  (evidence for relations)
    * ``(e:Entity)-[:HAS_GEOGRAPHY]->(g:Geography)``
    * ``(e:Entity)-[:PUBLISHED_IN_YEAR]->(y:Year)``
    * ``(e:Entity)-[:HAS_NUMERIC_VALUE]->(n:NumericValue)``
    * domain relations (uses_material, …) are loaded as their own
      type, e.g. ``(s:Material)-[:USES_MATERIAL]->(t:Process)``
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any, Sequence

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation
from ontology import EntityLabel, EntityType, NodeLabel, label_for

from .neo4j_config import Neo4jConfig

log = logging.getLogger(__name__)


# The set of node labels we expect to create constraints for.
# ``Entity`` is the legacy flat label kept for backward compatibility
# with previously-loaded data.
CONSTRAINT_LABELS: tuple[EntityLabel, ...] = tuple(EntityLabel)


class Neo4jLoader:
    """Async loader for the Nornikel Knowledge Graph."""

    def __init__(self, config: Neo4jConfig):
        try:
            from neo4j import AsyncGraphDatabase
        except Exception as exc:
            raise RuntimeError(
                "Neo4j driver is not installed. Install the optional 'neo4j' package "
                "before using Neo4jLoader."
            ) from exc

        self.config = config
        self.driver = AsyncGraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )

    async def close(self):
        await self.driver.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ---------------------------------------------------------------- schema

    async def setup_constraints(self) -> None:
        """Create uniqueness constraints and indexes for every entity label.

        Safe to call multiple times — uses ``IF NOT EXISTS`` everywhere.
        """
        statements: list[str] = []
        for label in EntityLabel:
            statements.append(
                f"CREATE CONSTRAINT {label.value.lower()}_name IF NOT EXISTS "
                f"FOR (n:{label.value}) REQUIRE n.name IS UNIQUE"
            )
        statements.append(
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) "
            "REQUIRE d.id IS UNIQUE"
        )
        statements.append(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) "
            "REQUIRE c.id IS UNIQUE"
        )
        statements.append(
            "CREATE CONSTRAINT alias_name IF NOT EXISTS FOR (a:Alias) "
            "REQUIRE a.name IS UNIQUE"
        )
        statements.append(
            "CREATE CONSTRAINT geo_name IF NOT EXISTS FOR (g:Geography) "
            "REQUIRE g.name IS UNIQUE"
        )
        statements.append(
            "CREATE CONSTRAINT year_value IF NOT EXISTS FOR (y:Year) "
            "REQUIRE y.value IS UNIQUE"
        )
        statements.append(
            "CREATE CONSTRAINT numeric_id IF NOT EXISTS FOR (n:NumericValue) "
            "REQUIRE n.id IS UNIQUE"
        )

        # Indexes for fast property filtering.
        index_statements = [
            "CREATE INDEX doc_source IF NOT EXISTS FOR (d:Document) ON (d.source)",
            "CREATE INDEX year_value_idx IF NOT EXISTS FOR (y:Year) ON (y.value)",
            "CREATE INDEX numeric_value_idx IF NOT EXISTS FOR (n:NumericValue) ON (n.numeric_value)",
            "CREATE INDEX numeric_unit_idx IF NOT EXISTS FOR (n:NumericValue) ON (n.unit)",
            "CREATE INDEX geo_kind IF NOT EXISTS FOR (g:Geography) ON (g.kind)",
        ]

        async with self.driver.session() as session:
            for stmt in statements + index_statements:
                await session.run(stmt)
        log.info("Neo4j constraints and indexes verified (%d total).", len(statements) + len(index_statements))

    # ---------------------------------------------------------------- entities

    async def load_entities(self, entities: Sequence[EnrichedEntity]) -> int:
        if not entities:
            return 0
        query = """
        UNWIND $batch AS row
        CALL apoc.merge.node(
            [row.label],
            {name: row.name},
            {attributes_json: row.attributes_json,
             created_at: timestamp()},
            {}
        ) YIELD node
        WITH node, row
        SET node.type = row.canonical_type,
            node.confidence = row.confidence

        WITH node, row
        MERGE (d:Document {id: row.source_document})
        MERGE (c:Chunk {id: row.chunk_id})
        ON CREATE SET c.char_start = row.char_start, c.char_end = row.char_end
        MERGE (d)-[:HAS_CHUNK]->(c)

        MERGE (c)-[m:MENTIONS]->(node)
        SET m.confidence = row.confidence, m.quote = row.quote

        WITH node, row
        UNWIND row.mentions AS mention
        MERGE (a:Alias {name: mention})
        MERGE (node)-[:KNOWN_AS]->(a)

        WITH node, row
        CALL apoc.do.when(
            row.geography IS NOT NULL,
            'MERGE (g:Geography {name: row.geography})
             ON CREATE SET g.kind = row.geo_kind
             MERGE (node)-[:HAS_GEOGRAPHY]->(g)
             RETURN count(g) AS c',
            'RETURN 0 AS c',
            {node: node, row: row}
        ) YIELD value AS gv

        WITH node, row
        CALL apoc.do.when(
            row.year IS NOT NULL,
            'MERGE (y:Year {value: row.year})
             MERGE (node)-[:PUBLISHED_IN_YEAR]->(y)
             RETURN count(y) AS c',
            'RETURN 0 AS c',
            {node: node, row: row}
        ) YIELD value AS yv

        WITH node, row
        CALL apoc.do.when(
            row.numeric_value IS NOT NULL,
            'MERGE (n:NumericValue {id: row.numeric_id})
             ON CREATE SET n.numeric_value = row.numeric_value,
                           n.unit = row.unit,
                           n.operator = row.numeric_operator,
                           n.min_value = row.numeric_min,
                           n.max_value = row.numeric_max,
                           n.property_name = row.numeric_property
             MERGE (node)-[:HAS_NUMERIC_VALUE]->(n)
             RETURN count(n) AS c',
            'RETURN 0 AS c',
            {node: node, row: row}
        ) YIELD value AS nv
        RETURN count(node) AS loaded
        """

        # Build batch rows. Pull geography/year/numeric out of
        # ``attributes`` so the query stays typed.
        batch_data: list[dict[str, Any]] = []
        for ent in entities:
            data = ent.model_dump()
            attrs = data.pop("attributes", {}) or {}
            name = (
                data.get("name")
                or data.get("entity")
                or data.get("canonical_name")
                or data.get("quote")
            )
            if not isinstance(name, str) or not name.strip():
                log.warning(
                    "Skipping entity without a loadable name: chunk_id=%s local_id=%s type=%s",
                    data.get("chunk_id"),
                    data.get("local_id"),
                    data.get("type"),
                )
                continue
            data["name"] = name.strip()
            label = label_for(data.get("type", EntityType.SUBSTANCE)).value
            data["label"] = label
            data["canonical_type"] = str(data.get("type"))
            data["attributes_json"] = json.dumps(attrs, ensure_ascii=False)

            # Geography: stored as attributes.geography (string) +
            # attributes.geo_kind (string from GeographyKind vocabulary).
            geo = attrs.get("geography")
            data["geography"] = geo if isinstance(geo, str) and geo.strip() else None
            data["geo_kind"] = attrs.get("geo_kind")

            # Year: stored as attributes.year (int).
            year_val = attrs.get("year")
            data["year"] = year_val if isinstance(year_val, int) else None

            # Numeric: stored as attributes.value / unit / operator / min / max
            # / property_name. This is the schema for the NumericValue anchor.
            numeric_value = attrs.get("value") or attrs.get("numeric_value")
            if numeric_value is not None:
                try:
                    numeric_value = float(numeric_value)
                except (TypeError, ValueError):
                    numeric_value = None
            data["numeric_value"] = numeric_value
            data["unit"] = attrs.get("unit")
            data["numeric_operator"] = attrs.get("operator")
            data["numeric_min"] = attrs.get("min")
            data["numeric_max"] = attrs.get("max")
            data["numeric_property"] = attrs.get("property_name")
            data["numeric_id"] = (
                f"{data['chunk_id']}::{data['local_id']}::numeric"
                if numeric_value is not None
                else None
            )
            batch_data.append(data)

        loaded = 0
        async with self.driver.session() as session:
            for i in range(0, len(batch_data), self.config.batch_size):
                batch = batch_data[i:i + self.config.batch_size]
                result = await session.run(query, batch=batch)
                record = await result.single()
                if record is not None:
                    loaded += int(record["loaded"] or 0)
        log.info("Loaded %d entities with proper labels.", loaded)
        return loaded

    # -------------------------------------------------------------- relations

    async def load_relations(self, relations: Sequence[EnrichedRelation]) -> int:
        if not relations:
            return 0

        # Group by relation type so we can use the type as a Cypher label.
        by_type: dict[str, list[dict[str, Any]]] = {}
        for rel in relations:
            rel_type = str(rel.relation_type)
            row = rel.model_dump()
            row["source_label"] = label_for(row.get("source_entity_type", EntityType.SUBSTANCE)).value
            row["target_label"] = label_for(row.get("target_entity_type", EntityType.SUBSTANCE)).value
            by_type.setdefault(rel_type, []).append(row)

        total = 0
        for rel_type, batch_data in by_type.items():
            # Sanitise the relation type for use as a Cypher label.
            safe_type = _safe_cypher_rel_type(rel_type)
            query = f"""
            UNWIND $batch AS row
            MATCH (s {{name: row.source_entity}})
            MATCH (t {{name: row.target_entity}})
            CALL apoc.merge.relationship(
                s, $rel_type,
                {{}},
                {{note: row.note}},
                t, {{}}
            ) YIELD rel
            SET rel.confidence = row.confidence,
                rel.quote = row.quote,
                rel.chunk_id = row.chunk_id,
                rel.source_document = row.source_document,
                rel.extractor = row.extractor,
                rel.needs_review = row.needs_review
            WITH rel, row, s, t
            CALL apoc.merge.node(
                ['Chunk'],
                {{id: row.chunk_id}},
                {{char_start: row.char_start, char_end: row.char_end}},
                {{}}
            ) YIELD node AS c
            MERGE (c)-[:SUPPORTS_ENTITY]->(s)
            MERGE (c)-[:SUPPORTS_ENTITY]->(t)
            RETURN count(rel) AS loaded
            """
            async with self.driver.session() as session:
                for i in range(0, len(batch_data), self.config.batch_size):
                    batch = batch_data[i:i + self.config.batch_size]
                    result = await session.run(query, batch=batch, rel_type=safe_type)
                    record = await result.single()
                    if record is not None:
                        total += int(record["loaded"] or 0)
        log.info("Loaded %d relations across %d types.", total, len(by_type))
        return total

    # ----------------------------------------------------------- housekeeping

    async def count_by_label(self) -> dict[str, int]:
        """Return a dict ``{label_name: count}`` for diagnostic purposes."""
        counts: dict[str, int] = {}
        async with self.driver.session() as session:
            for label in EntityLabel:
                result = await session.run(
                    f"MATCH (n:{label.value}) RETURN count(n) AS c"
                )
                record = await result.single()
                counts[label.value] = int(record["c"]) if record else 0
        return counts


def _safe_cypher_rel_type(name: str) -> str:
    """Make ``name`` a safe Cypher relationship type.

    Cypher relationship types must match ``[A-Z][A-Z0-9_]*``. We
    upper-case the input and replace any non-conforming characters
    with underscores.
    """
    out = []
    for i, ch in enumerate(name):
        if ch.isalnum() or ch == "_":
            out.append(ch.upper())
        else:
            out.append("_")
    if not out or not out[0].isalpha():
        out = ["R"] + out
    return "".join(out)


__all__ = ["Neo4jLoader", "CONSTRAINT_LABELS"]
