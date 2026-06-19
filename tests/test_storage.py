"""Repository and endpoint tests using SQLite in-memory — no Docker required."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from app.storage import repositories
from app.storage.db import get_db_session
from app.storage.orm import Base

_SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def db_session():
    """In-memory SQLite session with all tables created fresh per test."""
    engine = create_async_engine(_SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
async def api_client(db_session: AsyncSession):
    """FastAPI test client with DB dependency overridden to use SQLite."""
    async def _override():
        yield db_session

    app.dependency_overrides[get_db_session] = _override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Repository layer
# ---------------------------------------------------------------------------

async def test_create_and_get_session(db_session: AsyncSession) -> None:
    debate = await repositories.create_session(
        db=db_session,
        session_id="test-id-1",
        prompt="Is a hot dog a sandwich?",
        max_rounds=2,
        config={"participants": ["mock"]},
    )
    assert debate.id == "test-id-1"
    assert debate.status == "queued"

    fetched = await repositories.get_session(db_session, "test-id-1")
    assert fetched is not None
    assert fetched.prompt == "Is a hot dog a sandwich?"


async def test_get_session_returns_none_for_missing(db_session: AsyncSession) -> None:
    result = await repositories.get_session(db_session, "nonexistent")
    assert result is None


async def test_update_session_status(db_session: AsyncSession) -> None:
    await repositories.create_session(
        db=db_session,
        session_id="test-id-2",
        prompt="Test prompt",
        max_rounds=1,
        config={},
    )
    await repositories.update_session_status(db_session, "test-id-2", "running")
    fetched = await repositories.get_session(db_session, "test-id-2")
    assert fetched is not None
    assert fetched.status == "running"


# ---------------------------------------------------------------------------
# API endpoints (real DB round-trip via SQLite)
# ---------------------------------------------------------------------------

async def test_post_debate_persists_session(
    api_client: AsyncClient,
    db_session: AsyncSession,
    api_key_header: dict[str, str],
) -> None:
    response = await api_client.post(
        "/v1/debates",
        json={"prompt": "What is 2+2?"},
        headers=api_key_header,
    )
    assert response.status_code == 202
    session_id = response.json()["session_id"]

    saved = await repositories.get_session(db_session, session_id)
    assert saved is not None
    assert saved.prompt == "What is 2+2?"
    assert saved.status == "queued"


async def test_get_debate_returns_session(
    api_client: AsyncClient,
    api_key_header: dict[str, str],
) -> None:
    create_resp = await api_client.post(
        "/v1/debates",
        json={"prompt": "Hello?"},
        headers=api_key_header,
    )
    session_id = create_resp.json()["session_id"]

    get_resp = await api_client.get(f"/v1/debates/{session_id}", headers=api_key_header)
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["session_id"] == session_id
    assert body["prompt"] == "Hello?"
    assert body["status"] == "queued"


async def test_get_debate_returns_404_for_missing(
    api_client: AsyncClient,
    api_key_header: dict[str, str],
) -> None:
    response = await api_client.get("/v1/debates/no-such-id", headers=api_key_header)
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Repository: participant, answer, critique+scores, judgment
# ---------------------------------------------------------------------------

async def test_save_participant(db_session: AsyncSession) -> None:
    await repositories.create_session(
        db=db_session, session_id="s1", prompt="p", max_rounds=1, config={}
    )
    participant = await repositories.save_participant(
        db=db_session, session_id="s1",
        provider="mock", model="mock/default", role="debater",
    )
    assert participant.session_id == "s1"
    assert participant.role == "debater"


async def test_save_answer(db_session: AsyncSession) -> None:
    await repositories.create_session(
        db=db_session, session_id="s2", prompt="p", max_rounds=1, config={}
    )
    participant = await repositories.save_participant(
        db=db_session, session_id="s2",
        provider="mock", model="mock/default", role="debater",
    )
    answer = await repositories.save_answer(
        db=db_session,
        session_id="s2",
        participant_id=participant.id,
        round_number=0,
        content="My answer.",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=200,
    )
    assert answer.round_number == 0
    assert answer.content == "My answer."


async def test_save_critique_and_scores(db_session: AsyncSession) -> None:
    await repositories.create_session(
        db=db_session, session_id="s3", prompt="p", max_rounds=2, config={}
    )
    reviewer = await repositories.save_participant(
        db=db_session, session_id="s3",
        provider="mock", model="mock/default", role="debater",
    )
    target_participant = await repositories.save_participant(
        db=db_session, session_id="s3",
        provider="mock", model="mock/default", role="debater",
    )
    target_answer = await repositories.save_answer(
        db=db_session, session_id="s3", participant_id=target_participant.id,
        round_number=0, content="Target answer.", prompt_tokens=5,
        completion_tokens=5, latency_ms=100,
    )
    critique = await repositories.save_critique(
        db=db_session,
        session_id="s3",
        round_number=1,
        reviewer_id=reviewer.id,
        target_answer_id=target_answer.id,
        content="Good but incomplete.",
        scores=[{"dimension": "correctness", "value": 8, "rationale": "Mostly right."}],
    )
    assert critique.content == "Good but incomplete."


async def test_save_judgment(db_session: AsyncSession) -> None:
    await repositories.create_session(
        db=db_session, session_id="s4", prompt="p", max_rounds=1, config={}
    )
    judgment = await repositories.save_judgment(
        db=db_session,
        session_id="s4",
        final_answer="The answer is 42.",
        reasoning_summary="All models agreed.",
        confidence_score=9.0,
        strongest_contributions={},
        detected_disagreements={},
        corrected_mistakes={},
        score_summary={},
        caveats={},
        verification_status="verified",
    )
    assert judgment.final_answer == "The answer is 42."
