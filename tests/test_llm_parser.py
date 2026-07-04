"""Unit tests for the Yandex AI extraction pipeline (no real API calls)."""

import json
import os
import unittest
from unittest.mock import patch

from llm_pipeline_fewshot.llm_parser import (
    ChunkExtractor,
    MockLLMClient,
    create_llm_client,
    extract_json_object,
    make_chunk_input,
    parse_llm_json,
    _validate_entity,
    _validate_relations,
)
from llm_pipeline_fewshot.models import (
    LLMExtraction,
    LLMEntity,
    LLMRelation,
    RelationType,
)


class JsonExtractionTest(unittest.TestCase):
    def test_extract_plain_json(self):
        text = '{"entities": [], "relations": []}'
        self.assertEqual(extract_json_object(text), text)

    def test_extract_from_fenced_block(self):
        text = 'Some intro text.\n```json\n{"entities": [], "relations": []}\n```\nMore text.'
        result = extract_json_object(text)
        self.assertEqual(json.loads(result), {"entities": [], "relations": []})

    def test_extract_unfenced_json_with_prose(self):
        text = 'Here is the result: {"entities": [{"a": 1}], "relations": []} - end.'
        result = extract_json_object(text)
        obj = json.loads(result)
        self.assertEqual(obj["entities"], [{"a": 1}])

    def test_extract_handles_nested_braces(self):
        text = '{"entities": [{"a": {"b": 1}}], "relations": []}'
        self.assertEqual(extract_json_object(text), text)

    def test_extract_raises_on_no_json(self):
        with self.assertRaises(ValueError):
            extract_json_object("no json here")

    def test_parse_llm_json(self):
        text = '{"entities": [{"local_id": "e1", "type": "Material", "canonical_name": "никель", "mentions": ["никель"]}], "relations": []}'
        result = parse_llm_json(text)
        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.entities[0].local_id, "e1")


class EntityValidationTest(unittest.TestCase):
    def test_mention_not_in_text_is_warned(self):
        text = "Флотация меди при pH 10."
        raw = {
            "local_id": "e1",
            "type": "Process",
            "canonical_name": "Флотация",
            "mentions": ["Флотация", "вымышленное слово"],
        }
        cleaned, issues = _validate_entity(raw, text)
        self.assertTrue(any(issue.code == "mention_not_in_text" for issue in issues))

    def test_property_without_unit_is_warned(self):
        text = "Содержание серы 0,05."
        raw = {
            "local_id": "p1",
            "type": "Property",
            "canonical_name": "содержание серы",
            "mentions": ["Содержание серы"],
            "attributes": {"value_raw": "0,05"},
        }
        _, issues = _validate_entity(raw, text)
        self.assertTrue(any(issue.code == "property_value_no_unit" for issue in issues))


class RelationValidationTest(unittest.TestCase):
    def test_unknown_predicate_is_warned(self):
        issues = _validate_relations(
            [{"subject": "e1", "predicate": "made_up", "object": "e2"}],
            {"e1", "e2"},
        )
        self.assertTrue(any(issue.code == "unknown_predicate" for issue in issues))

    def test_missing_endpoint_is_error(self):
        issues = _validate_relations(
            [{"subject": "e1", "predicate": "uses_material", "object": "eX"}],
            {"e1"},
        )
        self.assertTrue(any(issue.code == "relation_endpoint_missing" for issue in issues))


class MockClientTest(unittest.TestCase):
    def test_mock_client_returns_valid_extraction_without_tokens(self):
        text = (
            "Сорбционная очистка от свинца выполнялась с использованием "
            "анионита Lewatit А365 при pH 3,0–3,5."
        )
        chunk = make_chunk_input(
            chunk_id="mock-1",
            text=text,
            source_document="doc.md",
        )
        extractor = ChunkExtractor(client=MockLLMClient())

        result = extractor.extract_chunk(chunk)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.model_uri, "mock://ner-re-examples")
        self.assertEqual(result.usage["total_tokens"], 0)
        self.assertGreaterEqual(len(result.entities), 2)
        self.assertTrue(any(entity.entity == "анионит Lewatit А365" for entity in result.entities))
        self.assertTrue(any(relation.relation_type == RelationType.USES_MATERIAL for relation in result.relations))

    def test_create_llm_client_env_switches_to_mock(self):
        with patch.dict(os.environ, {"LLM_CLIENT_MODE": "mock"}, clear=False):
            self.assertIsInstance(create_llm_client(), MockLLMClient)


class MakeChunkInputTest(unittest.TestCase):
    def test_builds_chunk_input(self):
        chunk = make_chunk_input(
            chunk_id="c1",
            text="Флотация меди при pH 10.",
            source_document="doc.md",
            char_start=100,
        )
        self.assertEqual(chunk.chunk_id, "c1")
        self.assertEqual(chunk.provenance.char_start, 100)
        self.assertEqual(chunk.provenance.char_end, 100 + len("Флотация меди при pH 10."))


if __name__ == "__main__":
    unittest.main()
