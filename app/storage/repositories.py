"""All database reads and writes live here. Nothing else touches SQLAlchemy queries."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.orm import Answer, Critique, DebateSession, Judgment, Participant, Score


async def create_session(
    db: AsyncSession,
    session_id: str,
    prompt: str,
    max_rounds: int,
    config: dict[str, Any],
) -> DebateSession:
    """
    Action: Insert a new debate_sessions row and return it.
    Trigger: Called by POST /v1/debates immediately after request validation.
    Arguments:
        db: Active async session from get_db_session().
        session_id: Pre-generated UUID string.
        prompt: Raw user prompt text.
        max_rounds: From the request config.
        config: Full config dict stored as JSONB for later reference.
    Output: The persisted DebateSession ORM object.
    """
    debate = DebateSession(
        id=session_id,
        prompt=prompt,
        max_rounds=max_rounds,
        config=config,
        status="queued",
    )
    db.add(debate)
    await db.commit()
    await db.refresh(debate)
    return debate


async def get_session(db: AsyncSession, session_id: str) -> DebateSession | None:
    """
    Action: Fetch one debate session by id, or None if not found.
    Trigger: Called by GET /v1/debates/{id}.
    Arguments:
        db: Active async session.
        session_id: UUID string from the URL path.
    Output: DebateSession ORM object or None.
    """
    result = await db.execute(select(DebateSession).where(DebateSession.id == session_id))
    return result.scalar_one_or_none()


async def update_session_status(
    db: AsyncSession, session_id: str, status: str
) -> None:
    """
    Action: Set the status field on a debate session.
    Trigger: Called by the worker at each phase transition (running, completed, failed).
    Arguments:
        db: Active async session.
        session_id: Target session UUID.
        status: New status string.
    Output: None (side-effect only).
    """
    debate = await get_session(db, session_id)
    if debate is None:
        raise ValueError(f"Session {session_id} not found")
    debate.status = status
    await db.commit()


async def save_participant(
    db: AsyncSession,
    session_id: str,
    provider: str,
    model: str,
    role: str,
) -> Participant:
    """
    Action: Insert a participant row and return it.
    Trigger: Called by the worker when setting up a debate session (M4).
    Arguments:
        db: Active async session.
        session_id: Parent session UUID.
        provider: Vendor name e.g. 'openai'.
        model: Model name e.g. 'gpt-4o'.
        role: 'debater' or 'judge'.
    Output: The persisted Participant ORM object.
    """
    participant = Participant(
        session_id=session_id, provider=provider, model=model, role=role
    )
    db.add(participant)
    await db.commit()
    await db.refresh(participant)
    return participant


async def save_answer(
    db: AsyncSession,
    session_id: str,
    participant_id: str,
    round_number: int,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
) -> Answer:
    """
    Action: Insert an answer row and return it.
    Trigger: Called by the worker after each provider.generate() call (M4).
    Arguments:
        db: Active async session.
        session_id: Parent session UUID.
        participant_id: Which model produced this answer.
        round_number: 0 for initial answers, 1+ for critique rounds.
        content: The model's text output.
        prompt_tokens / completion_tokens / latency_ms: From LLMResponse.
    Output: The persisted Answer ORM object.
    """
    answer = Answer(
        session_id=session_id,
        participant_id=participant_id,
        round_number=round_number,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )
    db.add(answer)
    await db.commit()
    await db.refresh(answer)
    return answer


async def save_critique(
    db: AsyncSession,
    session_id: str,
    reviewer_id: str,
    target_answer_id: str,
    round_number: int,
    content: str,
    scores: list[dict[str, Any]],
) -> Critique:
    """
    Action: Insert a critique + its score rows in one transaction.
    Trigger: Called by the worker after each peer-review step (M5).
    Arguments:
        db: Active async session.
        session_id: Parent session UUID.
        reviewer_id: Participant who wrote the critique.
        target_answer_id: Answer being critiqued.
        round_number: Which debate round this belongs to.
        content: Critique text.
        scores: List of {dimension, value, rationale} dicts (9 dimensions).
    Output: The persisted Critique ORM object (with scores attached).
    """
    critique = Critique(
        session_id=session_id,
        reviewer_id=reviewer_id,
        target_answer_id=target_answer_id,
        round_number=round_number,
        content=content,
    )
    db.add(critique)
    await db.flush()  # get critique.id before inserting scores

    target_answer = await db.get(Answer, target_answer_id)
    assert target_answer is not None
    for score_data in scores:
        db.add(
            Score(
                critique_id=critique.id,
                reviewer_id=reviewer_id,
                target_participant_id=target_answer.participant_id,
                dimension=score_data["dimension"],
                value=float(score_data["value"]),
                rationale=score_data.get("rationale", ""),
            )
        )
    await db.commit()
    await db.refresh(critique)
    return critique


async def save_judgment(
    db: AsyncSession,
    session_id: str,
    final_answer: str,
    reasoning_summary: str,
    strongest_contributions: dict[str, Any],
    detected_disagreements: dict[str, Any],
    corrected_mistakes: dict[str, Any],
    score_summary: dict[str, Any],
    confidence_score: float,
    caveats: dict[str, Any],
    verification_status: str | None,
) -> Judgment:
    """
    Action: Insert the final judgment row and mark the session completed.
    Trigger: Called by the worker after the judge model finishes (M6).
    Arguments:
        db: Active async session.
        session_id: Parent session UUID.
        (remaining): Fields from JudgeResult domain model.
    Output: The persisted Judgment ORM object.
    """
    judgment = Judgment(
        session_id=session_id,
        final_answer=final_answer,
        reasoning_summary=reasoning_summary,
        strongest_contributions=strongest_contributions,
        detected_disagreements=detected_disagreements,
        corrected_mistakes=corrected_mistakes,
        score_summary=score_summary,
        confidence_score=confidence_score,
        caveats=caveats,
        verification_status=verification_status,
    )
    db.add(judgment)
    await update_session_status(db, session_id, "completed")
    await db.refresh(judgment)
    return judgment
