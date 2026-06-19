from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_ok(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200


def test_correlation_id_echoed(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.headers.get("X-Request-ID")
