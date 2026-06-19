from __future__ import annotations

from fastapi.testclient import TestClient


def test_create_debate_requires_api_key(client: TestClient) -> None:
    response = client.post("/v1/debates", json={"prompt": "hi"})
    assert response.status_code == 401


def test_create_debate_returns_session_id(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response = client.post("/v1/debates", json={"prompt": "hi"}, headers=api_key_header)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["session_id"]


def test_create_debate_rejects_empty_prompt(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response = client.post("/v1/debates", json={"prompt": ""}, headers=api_key_header)
    assert response.status_code == 422
