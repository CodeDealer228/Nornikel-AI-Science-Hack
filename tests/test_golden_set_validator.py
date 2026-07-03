import copy
import unittest

from golden_set.validate import validate_sample


def valid_sample():
    text = "Флотация медной руды повышает извлечение меди при pH 10."
    return {
        "sample_id": "test_001",
        "chunk_id": "doc.md#0001",
        "document_id": "doc.md",
        "source_path": "Статьи/doc.md",
        "text": text,
        "entities": [
            {
                "id": "e1",
                "type": "Process",
                "canonical_name": "флотация",
                "mentions": [{"text": "Флотация", "start": 0, "end": 8}],
            },
            {
                "id": "e2",
                "type": "Material",
                "canonical_name": "медная руда",
                "mentions": [{"text": "медной руды", "start": 9, "end": 20}],
            },
            {
                "id": "e3",
                "type": "Property",
                "canonical_name": "извлечение меди",
                "mentions": [{"text": "извлечение меди", "start": 30, "end": 45}],
            },
        ],
        "relations": [
            {
                "id": "r1",
                "subject": "e1",
                "predicate": "uses_material",
                "object": "e2",
                "evidence_text": "Флотация медной руды",
                "evidence_start": 0,
                "evidence_end": 20,
            },
            {
                "id": "r2",
                "subject": "e1",
                "predicate": "affects_property",
                "object": "e3",
                "evidence_text": "Флотация медной руды повышает извлечение меди",
                "evidence_start": 0,
                "evidence_end": 45,
            },
        ],
    }


class GoldenSetValidatorTest(unittest.TestCase):
    def assert_has_error(self, sample, expected):
        errors = validate_sample(sample, 1)
        self.assertTrue(
            any(expected in error for error in errors),
            f"Expected {expected!r} in errors: {errors}",
        )

    def test_valid_sample_passes(self):
        self.assertEqual(validate_sample(valid_sample(), 1), [])

    def test_unknown_entity_type_fails(self):
        sample = valid_sample()
        sample["entities"][0]["type"] = "UnknownType"
        self.assert_has_error(sample, "entity[0].type")

    def test_unknown_relation_predicate_fails(self):
        sample = valid_sample()
        sample["relations"][0]["predicate"] = "unknown_predicate"
        self.assert_has_error(sample, "relation[0].predicate")

    def test_relation_unknown_subject_fails(self):
        sample = valid_sample()
        sample["relations"][0]["subject"] = "missing"
        self.assert_has_error(sample, "relation[0].subject")

    def test_mention_text_not_in_sample_fails(self):
        sample = valid_sample()
        sample["entities"][0]["mentions"][0]["text"] = "Пирометаллургия"
        self.assert_has_error(sample, "mention")

    def test_mention_offsets_mismatch_fails(self):
        sample = valid_sample()
        sample["entities"][0]["mentions"][0]["start"] = 1
        self.assert_has_error(sample, "slice does not match mention.text")

    def test_evidence_text_not_in_sample_fails(self):
        sample = valid_sample()
        sample["relations"][0]["evidence_text"] = "несуществующая цитата"
        self.assert_has_error(sample, "evidence_text is not an exact substring")

    def test_evidence_offsets_mismatch_fails(self):
        sample = valid_sample()
        sample["relations"][0]["evidence_start"] = 1
        self.assert_has_error(sample, "slice does not match evidence_text")

    def test_mutation_guard(self):
        sample = valid_sample()
        before = copy.deepcopy(sample)
        validate_sample(sample, 1)
        self.assertEqual(sample, before)


if __name__ == "__main__":
    unittest.main()
