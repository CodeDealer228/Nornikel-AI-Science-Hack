import unittest

from ensemble import EnsembleMerger
from graph_reasoning import GraphEdge, GraphNode, GraphReasoner, GraphReasoningContext
from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation, EntityType, RelationType
from quality_control import FactQualityController
from routing import ExtractionRoute, ExtractionRouter


def entity(local_id, name, extractor, confidence=0.8, entity_type=EntityType.MATERIAL):
    return EnrichedEntity(
        entity=name,
        type=entity_type,
        chunk_id="c1",
        source_document="doc.md",
        quote=name,
        confidence=confidence,
        local_id=local_id,
        mentions=[name],
        char_start=0,
        char_end=len(name),
        extractor=extractor,
    )


def relation(source, target, predicate=RelationType.USES_MATERIAL, confidence=0.75):
    return EnrichedRelation(
        source_entity=source.entity,
        target_entity=target.entity,
        relation_type=predicate,
        chunk_id="c1",
        source_document="doc.md",
        quote=f"{source.entity} -> {target.entity}",
        confidence=confidence,
        source_entity_type=source.type,
        target_entity_type=target.type,
        source_local_id=source.local_id,
        target_local_id=target.local_id,
        char_start=0,
        char_end=10,
        extractor="yandex_llm",
    )


class EnsembleRoutingGraphQualityTest(unittest.TestCase):
    def test_ensemble_merges_same_entity_from_two_sources(self):
        n = entity("n1", "медная руда", "natasha", 0.55)
        l = entity("l1", "Медная руда", "yandex_llm", 0.86)
        result = EnsembleMerger().merge([n], [l])
        self.assertEqual(len(result.entities), 1)
        self.assertGreater(result.entities[0].confidence, 0.8)
        self.assertEqual(result.entity_decisions[0].sources, ("natasha", "yandex_llm"))

    def test_router_selects_ensemble_for_technical_relation_dense_chunk(self):
        decision = ExtractionRouter().route_chunk(
            "Флотация повышает извлечение меди до 92 % и зависит от pH 10.",
            natasha_entities=[entity("n1", "Флотация", "natasha", entity_type=EntityType.PROCESS)],
        )
        self.assertEqual(decision.route, ExtractionRoute.ENSEMBLE)

    def test_graph_reasoner_detects_explicit_contradiction_and_gap(self):
        context = GraphReasoningContext(
            seed_entities=("A",),
            nodes=[
                GraphNode(id="a", name="A", type="Process", confidence=0.9),
                GraphNode(id="b", name="B", type="Conclusion", confidence=0.4),
            ],
            edges=[
                GraphEdge("a", "b", "contradicts", quote="A contradicts B", confidence=0.8),
            ],
        )
        enriched = GraphReasoner().enrich_context(context)
        self.assertEqual(len(enriched.contradictions), 1)
        self.assertTrue(any(gap.code == "low_confidence_entity" for gap in enriched.gaps))

    def test_quality_controller_detects_missing_relation_endpoint(self):
        source = entity("e1", "процесс", "yandex_llm", entity_type=EntityType.PROCESS)
        target = entity("missing", "руда", "yandex_llm")
        rel = relation(source, target)
        report = FactQualityController().inspect([source], [rel])
        self.assertTrue(report.has_errors)
        self.assertTrue(any(issue.code == "missing_relation_endpoint" for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
