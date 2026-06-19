"""Async SQLAlchemy engine + session factory. One engine per process."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

_settings = get_settings()
_engine = create_async_engine(_settings.database_url, pool_pre_ping=True)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields one session per request, auto-closes after."""
    async with _session_factory() as session:
        yield session


async def check_db() -> bool:
    """Return True if Postgres is reachable. Used by /readyz."""
    try:
        async with _session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
