"""Liveness and readiness endpoints. No auth (used by orchestrators/load balancers)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.storage.db import check_db

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, str]:
    """Readiness: Postgres is reachable. Redis check added in M4."""
    if not await check_db():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="db unavailable"
        )
    return {"status": "ready"}
