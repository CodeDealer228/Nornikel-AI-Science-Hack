import json
import logging
from typing import List

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation

from .neo4j_config import Neo4jConfig

log = logging.getLogger(__name__)


class Neo4jLoader:
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

    async def setup_constraints(self):
        queries = [
            "CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            "CREATE CONSTRAINT alias_name IF NOT EXISTS FOR (a:Alias) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        ]
        async with self.driver.session() as session:
            for query in queries:
                await session.run(query)
        log.info("Neo4j constraints verified.")

    async def load_entities(self, entities: List[EnrichedEntity]):
        if not entities:
            return

        query = """
        UNWIND $batch AS row
        MERGE (e:Entity {name: row.entity, type: row.type})
        ON CREATE SET e.created_at = timestamp()
        SET e.attributes_json = row.attributes_json

        WITH e, row
        MERGE (d:Document {id: row.source_document})
        MERGE (c:Chunk {id: row.chunk_id})
        ON CREATE SET c.char_start = row.char_start, c.char_end = row.char_end
        MERGE (d)-[:HAS_CHUNK]->(c)

        MERGE (c)-[m:MENTIONS]->(e)
        SET m.confidence = row.confidence, m.quote = row.quote

        WITH e, row
        UNWIND row.mentions AS mention
        MERGE (a:Alias {name: mention})
        MERGE (e)-[:KNOWN_AS]->(a)
        """

        batch_data = []
        for ent in entities:
            data = ent.model_dump()
            data.pop("attributes", None)
            data["type"] = str(data.get("type"))
            data["attributes_json"] = json.dumps(ent.attributes, ensure_ascii=False)
            batch_data.append(data)

        for i in range(0, len(batch_data), self.config.batch_size):
            batch = batch_data[i:i + self.config.batch_size]
            async with self.driver.session() as session:
                await session.run(query, batch=batch)

    async def load_relations(self, relations: List[EnrichedRelation]):
        if not relations:
            return

        by_type = {}
        for rel in relations:
            rel_type = str(rel.relation_type)
            by_type.setdefault(rel_type, []).append(rel.model_dump())

        for rel_type, batch_data in by_type.items():
            query = f"""
            UNWIND $batch AS row
            MATCH (s:Entity {{name: row.source_entity, type: row.source_entity_type}})
            MATCH (t:Entity {{name: row.target_entity, type: row.target_entity_type}})
            MERGE (s)-[r:{rel_type}]->(t)
            ON CREATE SET r.created_at = timestamp()
            SET r.note = row.note

            WITH r, row
            MATCH (c:Chunk {{id: row.chunk_id}})
            MERGE (c)-[sup:SUPPORTS]->(r)
            SET sup.confidence = row.confidence, sup.quote = row.quote
            """

            for i in range(0, len(batch_data), self.config.batch_size):
                batch = batch_data[i:i + self.config.batch_size]
                async with self.driver.session() as session:
                    await session.run(query, batch=batch)
