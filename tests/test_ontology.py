"""Tests for the new ontology types and per-type label mapping."""

import unittest

from ontology import (
    ENTITY_TYPE_TO_LABEL,
    EntityLabel,
    EntityType,
    GeographyKind,
    NumericOperator,
    RelationType,
    label_for,
)


class OntologyTest(unittest.TestCase):
    def test_all_entity_types_have_labels(self):
        for etype in EntityType:
            self.assertIn(etype, ENTITY_TYPE_TO_LABEL)
            self.assertEqual(ENTITY_TYPE_TO_LABEL[etype].value, etype.value)

    def test_label_for_returns_label(self):
        self.assertEqual(label_for(EntityType.MATERIAL), EntityLabel.MATERIAL)
        self.assertEqual(label_for("Process"), EntityLabel.PROCESS)
        self.assertEqual(label_for(EntityType.GEOGRAPHY), EntityLabel.GEOGRAPHY)
        self.assertEqual(label_for(EntityType.YEAR), EntityLabel.YEAR)
        self.assertEqual(label_for(EntityType.NUMERIC_VALUE), EntityLabel.NUMERIC_VALUE)

    def test_new_relation_types_present(self):
        # Relations introduced in the schema update.
        self.assertEqual(RelationType.HAS_GEOGRAPHY.value, "has_geography")
        self.assertEqual(RelationType.PUBLISHED_IN_YEAR.value, "published_in_year")
        self.assertEqual(RelationType.HAS_NUMERIC_VALUE.value, "has_numeric_value")

    def test_geography_vocabulary(self):
        self.assertIn(GeographyKind.RUSSIA, GeographyKind)
        self.assertIn(GeographyKind.WORLDWIDE, GeographyKind)
        self.assertIn(GeographyKind.UNKNOWN, GeographyKind)

    def test_numeric_operator_vocabulary(self):
        self.assertEqual(NumericOperator.LTE, "<=")
        self.assertEqual(NumericOperator.GTE, ">=")
        self.assertEqual(NumericOperator.RANGE, "range")


if __name__ == "__main__":
    unittest.main()
