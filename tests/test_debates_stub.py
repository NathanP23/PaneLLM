from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.main import app
from app.storage.db import get_db_session
from app.storage.orm import Base

_SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(autouse=True)
async def override_db():
    """Replace the real DB dependency with SQLite for every test in this module."""
    engine = create_async_engine(_SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_test_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _get_test_session
    yield
    app.dependency_overrides.clear()
    await engine.dispose()


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
