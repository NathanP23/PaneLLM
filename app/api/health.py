"""Liveness and readiness endpoints. No auth (used by orchestrators/load balancers)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict[str, str]:
    """Readiness: dependencies are reachable. Checks Postgres/Redis once they land (M3/M4)."""
    # ponytail: no deps to check yet; extend when DB/Redis are wired.
    return {"status": "ready"}
