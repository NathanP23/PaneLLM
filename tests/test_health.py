from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_healthz_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_ok_when_db_up(client: TestClient) -> None:
    with patch("app.api.health.check_db", new=AsyncMock(return_value=True)):
        response = client.get("/readyz")
    assert response.status_code == 200


def test_readyz_503_when_db_down(client: TestClient) -> None:
    with patch("app.api.health.check_db", new=AsyncMock(return_value=False)):
        response = client.get("/readyz")
    assert response.status_code == 503


def test_correlation_id_echoed(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.headers.get("X-Request-ID")
