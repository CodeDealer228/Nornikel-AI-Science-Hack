from typing import List

from llm_pipeline_fewshot.models import EnrichedEntity, EnrichedRelation

from .evaluation_matcher import compare_entities, compare_relations
from .evaluation_metrics import EvaluationReport, MetricResult


class Evaluator:
    def __init__(self) -> None:
        self.micro_entity = MetricResult()
        self.micro_relation = MetricResult()

    def evaluate_chunk(
        self,
        pred_entities: List[EnrichedEntity],
        exp_entities: List[EnrichedEntity],
        pred_relations: List[EnrichedRelation],
        exp_relations: List[EnrichedRelation],
    ) -> EvaluationReport:
        e_tp, e_fp, e_fn = compare_entities(pred_entities, exp_entities)
        r_tp, r_fp, r_fn = compare_relations(pred_relations, exp_relations)

        self.micro_entity.true_positives += e_tp
        self.micro_entity.false_positives += e_fp
        self.micro_entity.false_negatives += e_fn

        self.micro_relation.true_positives += r_tp
        self.micro_relation.false_positives += r_fp
        self.micro_relation.false_negatives += r_fn

        return EvaluationReport(
            entity_metrics=MetricResult(e_tp, e_fp, e_fn),
            relation_metrics=MetricResult(r_tp, r_fp, r_fn),
        )

    def get_micro_report(self) -> EvaluationReport:
        return EvaluationReport(
            entity_metrics=self.micro_entity,
            relation_metrics=self.micro_relation,
        )
