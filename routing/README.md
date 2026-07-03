# Routing — query-time decision engine

This package contains two routing layers:

* `ExtractionRouter` (`extraction_router.py`) — **extraction-time**
  routing. Decides *which extractor* to run on a chunk
  (Natasha / LLM / ensemble / skip). Already existed.
* `QueryRouter` (`query_router.py`) — **query-time** routing.
  Decides *which knowledge source* should answer a user query:
  `GRAPH_ONLY`, `RAG_ONLY`, `HYBRID`, or `NO_DATA`.

The query-time router is the final intelligence layer of the
Knowledge Graph pipeline. It does **not** generate answers and
does **not** implement RAG — it only emits a routing decision
and the signals that produced it.

## Pipeline

```
user query
   │
   ▼
QueryEntityExtractor  ──►  seed entities, query analysis
   │
   ▼
GraphCoverageAnalyzer ──►  GraphCoverageReport
   │                         (seed coverage, multi-hop,
   │                          relation density, contradictions,
   │                          knowledge gaps)
   ▼
QueryRouter            ──►  QueryRoutingDecision
                            { route, confidence,
                              coverage_score, ambiguity_score,
                              reasons, signals }
```

## Decision logic

The router combines a small set of rule-based gates (for the
extreme cases) with weighted scoring for everything in between.

### Coverage score (0..1)

| Component           | Default weight | Source                          |
|---------------------|---------------:|---------------------------------|
| Seed coverage       | 0.40           | `seed_coverage_ratio`           |
| Multi-hop reach     | 0.20           | `max_hop_observed` (0..4 hops)  |
| Relation density    | 0.20           | `log1p(edges/nodes)` (≤4)       |
| Source diversity    | 0.10           | `log1p(unique_docs)` (≤5)       |
| Avg edge confidence | 0.10           | `avg_edge_confidence`           |

A 10% penalty is applied when explicit `contradicts` relations
are present in the subgraph.

### Confidence score (0..1)

| Component           | Default weight | Source                          |
|---------------------|---------------:|---------------------------------|
| Avg edge confidence | 0.65           | `avg_edge_confidence`           |
| Inverse gap count   | 0.35           | `1 / (1 + gap_count)`           |

### Ambiguity score (0..1, lower = clearer)

Combines token count, seed entity count and a graph-clarity
bonus. Very short queries with no entities score near 1.0;
rich, well-covered queries score near 0.

### Rules (in order)

1. Empty / whitespace query → `NO_DATA`
2. Low-signal query (no seeds, no numeric constraints, very
   short) → `NO_DATA`
3. Graph empty AND no seed entities → `NO_DATA`
4. Graph empty but seed entities present → `RAG_ONLY`
5. Graph has matched nodes but no relations → `RAG_ONLY`
6. No seeds matched AND no edges → `RAG_ONLY`
7. Coverage ≥ 0.62 AND confidence ≥ 0.55 AND ambiguity ≤ 0.70
   AND no contradictions → `GRAPH_ONLY`
8. Coverage below `GRAPH_ONLY` threshold (or contradictions
   present) → `HYBRID`
9. Otherwise → `RAG_ONLY`

Thresholds are constructor parameters of `QueryRouter`.

### Explicit-marker overrides

Some query shapes are unambiguous regardless of the coverage
score. The router checks these markers **before** the generic
rule-gate and short-circuits when they match (first match wins):

| Marker | Trigger | Route | Reason |
|---|---|---|---|
| `is_definitional` | graph has < 2 edges | `RAG_ONLY` | graph stores facts, not definitions |
| `is_geo_comparison` | graph has ≥ 1 edge | `HYBRID` | need both structured facts and prose |
| `is_comparison` (not geo) | graph has ≥ 1 edge | `HYBRID` | need both structured facts and prose |
| `is_causal` | graph has ≥ 1 edge | `HYBRID` | explanations live in prose |
| `has_numeric_constraint` | graph has ≥ 3 edges and coverage is near the `GRAPH_ONLY` threshold | `GRAPH_ONLY` | numeric ranges are the graph's strong suit; only boosts |

The markers are detected by `QueryEntityExtractor.analyze` from
regex patterns in `routing/query_entity_extractor.py`. They
appear in `QueryAnalysis` as `is_definitional`, `is_causal`,
`is_comparison`, `is_geo_comparison`.

## Integration points

* Reuses `chunking.natasha_pipeline.get_pipeline` for primary
  NER on the query.
* Reuses `synonym_normalization.canonicalizer.canonicalize_text`
  and `SynonymDictionary` for alias resolution.
* Reuses `graph_reasoning.neo4j_subgraph.Neo4jSubgraphExtractor`
  for graph traversal (1–4 hops).
* Reuses `graph_reasoning.reasoner.GraphReasoner` for
  contradiction / knowledge-gap detection.
* The existing `ensemble.merger.EnsembleMerger` is **not**
  modified; its outputs are not consulted at query time.

## Quick start

```python
import asyncio
from neo4j_integration.neo4j_config import Neo4jConfig
from neo4j_integration.neo4j_loader import Neo4jLoader
from routing import build_query_router

async def main():
    config = Neo4jConfig()
    async with Neo4jLoader(config) as loader:
        router = build_query_router(driver=loader.driver)
        decision = await router.route(
            "Какие методы обессоливания воды подходят при "
            "сульфатах ≤ 300 мг/л?"
        )
        print(decision.route, decision.coverage_score, decision.reasons)

asyncio.run(main())
```

`build_query_router(driver=None)` is also available for offline
or test deployments — without a driver the graph coverage step
reports an empty subgraph, so the router gracefully degrades
to `RAG_ONLY` / `NO_DATA` based on the query structure alone.

## Files

| File                          | Purpose                                              |
|-------------------------------|------------------------------------------------------|
| `models.py`                   | Extraction-time routing models (unchanged)           |
| `extraction_router.py`        | Extraction-time routing logic (unchanged)            |
| `query_models.py`             | Query-time data classes                              |
| `query_entity_extractor.py`   | Natasha + regex query NER                            |
| `graph_coverage.py`           | Neo4j subgraph coverage summarisation                |
| `query_router.py`             | Decision engine                                      |
| `__init__.py`                 | Public exports                                       |
