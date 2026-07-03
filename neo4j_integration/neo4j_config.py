import os
from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    batch_size: int = int(os.getenv("NEO4J_BATCH_SIZE", "500"))
