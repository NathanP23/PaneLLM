"""Debate endpoints. M1: stub that validates input and returns a session id.

Real orchestration (enqueue + worker + persistence) lands in M3/M4.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

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


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_debate(request: CreateDebateRequest) -> CreateDebateResponse:
    """Accept a prompt and return a session id. Stub: nothing runs or persists yet."""
    # ponytail: in-memory id only; DB row + arq enqueue replace this in M3/M4.
    return CreateDebateResponse(session_id=str(uuid.uuid4()), status="queued")
