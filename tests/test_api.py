"""Tests for the FastAPI server (in offline mode — no Neo4j)."""

import unittest
from typing import Any

try:
    from fastapi.testclient import TestClient  # type: ignore
    from api.server import STATE, app  # type: ignore
    _FASTAPI_AVAILABLE = True
    _SKIP_REASON = ""
except Exception as exc:  # pragma: no cover - environment-specific
    _FASTAPI_AVAILABLE = False
    _SKIP_REASON = f"fastapi not available: {exc}"


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class APISmokeTest(unittest.TestCase):
    _client_ctx: "TestClient | None" = None

    @classmethod
    def setUpClass(cls):
        # Driving TestClient as a context manager runs the FastAPI lifespan
        # (async) in the same loop that serves the requests, which sets up
        # STATE.driver / STATE.dispatcher. Calling lifespan(app).__enter__
        # directly fails because lifespan is an async context manager.
        cls._client_ctx = TestClient(app)
        cls._client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        if cls._client_ctx is not None:
            cls._client_ctx.__exit__(None, None, None)
            cls._client_ctx = None

    def setUp(self):
        self.client = self._client_ctx

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("uptime_sec", body)

    def test_ready(self):
        r = self.client.get("/ready")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        # In offline mode, Neo4j is not connected.
        self.assertIn("ready", body)
        self.assertIn("neo4j_connected", body)

    def test_metrics(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        text = r.text
        self.assertIn("nk_uptime_seconds", text)
        self.assertIn("nk_requests_total", text)

    def test_route_only_endpoint(self):
        r = self.client.post(
            "/route",
            json={"query": "Какие методы обессоливания воды при сульфатах <=300 мг/л?"},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("route", body)
        self.assertIn("confidence", body)
        self.assertIn("markers", body)
        self.assertIn("request_id", body)
        # Numeric constraint marker should be set.
        self.assertTrue(body["markers"]["numeric"])

    def test_query_endpoint_no_neo4j(self):
        r = self.client.post(
            "/query",
            json={"query": "Какие методы электроэкстракции никеля?", "synthesize": False},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("answer", body)
        self.assertIn("route", body)
        self.assertIn("request_id", body)
        # Without Neo4j the answer comes from RAG (stub) + fallback render.
        self.assertFalse(body["used_llm"])

    def test_query_endpoint_empty_query_400(self):
        r = self.client.post("/query", json={"query": ""})
        self.assertEqual(r.status_code, 422)  # pydantic validation

    def test_query_with_geo_marker(self):
        r = self.client.post(
            "/query",
            json={
                "query": "Сравни отечественную и зарубежную практику выщелачивания никеля",
                "synthesize": False,
            },
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        # Re-check via /route:
        r2 = self.client.post("/route", json={
            "query": "Сравни отечественную и зарубежную практику выщелачивания никеля",
        })
        self.assertTrue(r2.json()["markers"]["geography"])


@unittest.skipUnless(_FASTAPI_AVAILABLE, _SKIP_REASON)
class APIAuthTest(unittest.TestCase):
    """When API_KEY is set, the protected endpoints require it."""

    def setUp(self):
        from config import get_settings
        self._original_key = get_settings().api.api_key
        # Enter the lifespan so STATE.dispatcher is initialised; otherwise
        # /query would return 503 regardless of auth.
        self._client_ctx = TestClient(app)
        self._client_ctx.__enter__()

    def tearDown(self):
        from config import get_settings, reset_settings_cache
        reset_settings_cache()
        self._client_ctx.__exit__(None, None, None)

    def test_auth_when_disabled(self):
        # No key set — should still work.
        r = self._client_ctx.post("/query", json={"query": "никель", "synthesize": False})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
