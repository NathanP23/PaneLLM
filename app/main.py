"""FastAPI entry point: wires config, logging, CORS, the API-key gate, and routers.

Orchestrator only — no business logic lives here.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.api import debates, health
from app.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger("app.main")

app = FastAPI(title="PaneLLM", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Bind a correlation id to every log line and echo it back as X-Request-ID."""
    correlation_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    clear_contextvars()
    bind_contextvars(correlation_id=correlation_id, path=request.url.path)
    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """API-key gate for non-public routes. Real auth comes later."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")


app.include_router(health.router)
app.include_router(debates.router, dependencies=[Depends(require_api_key)])
