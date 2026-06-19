"""Debate endpoints."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage import repositories
from app.storage.db import get_db_session

router = APIRouter(prefix="/v1/debates", tags=["debates"])


class DebateConfig(BaseModel):
    max_rounds: int = Field(default=1, ge=1, le=5)
    participants: list[str] = Field(default_factory=lambda: ["mock"])


class CreateDebateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    config: DebateConfig = Field(default_factory=DebateConfig)


class CreateDebateResponse(BaseModel):
    session_id: str
    status: str


class DebateSessionResponse(BaseModel):
    session_id: str
    status: str
    prompt: str
    max_rounds: int
    config: dict[str, Any]
    created_at: str
    completed_at: str | None


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_debate(
    request: CreateDebateRequest,
    db: AsyncSession = Depends(get_db_session),
) -> CreateDebateResponse:
    """Validate, persist, and enqueue a debate session."""
    # ponytail: enqueue (arq) replaces the stub in M4.
    session_id = str(uuid.uuid4())
    debate = await repositories.create_session(
        db=db,
        session_id=session_id,
        prompt=request.prompt,
        max_rounds=request.config.max_rounds,
        config=request.config.model_dump(),
    )
    return CreateDebateResponse(session_id=debate.id, status=debate.status)


@router.get("/{session_id}")
async def get_debate(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> DebateSessionResponse:
    """Fetch a debate session by id."""
    debate = await repositories.get_session(db, session_id)
    if debate is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")
    return DebateSessionResponse(
        session_id=debate.id,
        status=debate.status,
        prompt=debate.prompt,
        max_rounds=debate.max_rounds,
        config=debate.config,
        created_at=debate.created_at.isoformat(),
        completed_at=debate.completed_at.isoformat() if debate.completed_at else None,
    )
