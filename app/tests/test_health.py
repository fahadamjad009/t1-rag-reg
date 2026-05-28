"""Smoke test: /health endpoint returns expected shape."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "t1-rag-reg"


def test_debug_build_returns_config():
    r = client.get("/debug/build")
    assert r.status_code == 200
    body = r.json()
    assert "build_id" in body
    assert "calibration_ranges" in body
    assert isinstance(body["calibration_ranges"], dict)
