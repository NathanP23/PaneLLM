"""SQLAlchemy ORM table definitions. Single source of truth for the DB schema."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DebateSession(Base):
    __tablename__ = "debate_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued", index=True)
    max_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    participants: Mapped[list[Participant]] = relationship(back_populates="session")
    answers: Mapped[list[Answer]] = relationship(back_populates="session")
    judgment: Mapped[Judgment | None] = relationship(back_populates="session", uselist=False)


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("debate_sessions.id"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    # "debater" or "judge"
    role: Mapped[str] = mapped_column(String, nullable=False, default="debater")

    session: Mapped[DebateSession] = relationship(back_populates="participants")
    answers: Mapped[list[Answer]] = relationship(back_populates="participant")


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("debate_sessions.id"), nullable=False, index=True
    )
    participant_id: Mapped[str] = mapped_column(
        String, ForeignKey("participants.id"), nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    session: Mapped[DebateSession] = relationship(back_populates="answers")
    participant: Mapped[Participant] = relationship(back_populates="answers")
    critiques_given: Mapped[list[Critique]] = relationship(
        back_populates="target_answer", foreign_keys="Critique.target_answer_id"
    )


class Critique(Base):
    __tablename__ = "critiques"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("debate_sessions.id"), nullable=False, index=True
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewer_id: Mapped[str] = mapped_column(
        String, ForeignKey("participants.id"), nullable=False
    )
    target_answer_id: Mapped[str] = mapped_column(
        String, ForeignKey("answers.id"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    target_answer: Mapped[Answer] = relationship(
        back_populates="critiques_given", foreign_keys=[target_answer_id]
    )
    scores: Mapped[list[Score]] = relationship(back_populates="critique")


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    critique_id: Mapped[str] = mapped_column(
        String, ForeignKey("critiques.id"), nullable=False, index=True
    )
    reviewer_id: Mapped[str] = mapped_column(
        String, ForeignKey("participants.id"), nullable=False
    )
    target_participant_id: Mapped[str] = mapped_column(
        String, ForeignKey("participants.id"), nullable=False
    )
    # one of the 9 scoring dimensions
    dimension: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")

    critique: Mapped[Critique] = relationship(back_populates="scores")


class Judgment(Base):
    __tablename__ = "judgments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("debate_sessions.id"), nullable=False, unique=True
    )
    final_answer: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    strongest_contributions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    detected_disagreements: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    corrected_mistakes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    score_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    caveats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    verification_status: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    session: Mapped[DebateSession] = relationship(back_populates="judgment")
