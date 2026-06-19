"""structlog setup: JSON logs with a correlation id bound per request."""

from __future__ import annotations

import logging

import structlog


def configure_logging(log_level: str) -> None:
    """Configure structlog to emit JSON. Call once at app startup."""
    logging.basicConfig(format="%(message)s", level=log_level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
