"""Tests for centralized settings and structured logging."""

import io
import json
import logging
import unittest

from config import (
    APISettings,
    IngestSettings,
    LoggingSettings,
    Neo4jSettings,
    RAGSettings,
    RouterSettings,
    Settings,
    YandexGPTSettings,
    get_settings,
    reset_settings_cache,
)
from logging_setup import (
    JsonFormatter,
    configure_logging,
    get_logger,
    log_with,
    shutdown_logging,
    set_request_id,
)


class SettingsTest(unittest.TestCase):
    def test_get_settings_returns_cached_singleton(self):
        reset_settings_cache()
        a = get_settings()
        b = get_settings()
        self.assertIs(a, b)

    def test_settings_have_expected_subsettings(self):
        s = Settings()
        self.assertIsInstance(s.neo4j, Neo4jSettings)
        self.assertIsInstance(s.yandex_gpt, YandexGPTSettings)
        self.assertIsInstance(s.router, RouterSettings)
        self.assertIsInstance(s.ingest, IngestSettings)
        self.assertIsInstance(s.rag, RAGSettings)
        self.assertIsInstance(s.api, APISettings)
        self.assertIsInstance(s.logging, LoggingSettings)

    def test_yandex_gpt_is_configured(self):
        s = YandexGPTSettings(api_key="", folder_id="")
        self.assertFalse(s.is_configured)
        s2 = YandexGPTSettings(api_key="k", folder_id="f")
        self.assertTrue(s2.is_configured)

    def test_reset_settings_cache(self):
        reset_settings_cache()
        first = get_settings()
        reset_settings_cache()
        second = get_settings()
        # Should be a fresh instance, not the same object.
        self.assertIsNot(first, second)


class LoggingTest(unittest.TestCase):
    def setUp(self):
        shutdown_logging()

    def tearDown(self):
        shutdown_logging()

    def test_configure_logging_idempotent(self):
        configure_logging(level="INFO", fmt="text")
        configure_logging(level="DEBUG", fmt="text")
        logger = get_logger("test_logger")
        # After second configure, level should be DEBUG.
        self.assertEqual(logger.getEffectiveLevel(), logging.DEBUG)

    def test_text_formatter(self):
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        out = formatter.format(record)
        self.assertIn("hello", out)
        self.assertIn("INFO", out)

    def test_json_formatter(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0,
            msg="hi", args=(), exc_info=None,
        )
        out = formatter.format(record)
        data = json.loads(out)
        self.assertEqual(data["message"], "hi")
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "x")

    def test_request_id_attaches_to_json(self):
        set_request_id("req-123")
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        out = formatter.format(record)
        data = json.loads(out)
        self.assertEqual(data["request_id"], "req-123")

    def test_log_with_extra_fields(self):
        configure_logging(level="INFO", fmt="text")
        logger = get_logger("test_log_with")
        # Just confirm it doesn't crash.
        log_with(logger, logging.INFO, "graph_lookup", seeds=3, max_hop=4)
        log_with(logger, logging.WARNING, "coverage_low", score=0.42)


if __name__ == "__main__":
    unittest.main()
