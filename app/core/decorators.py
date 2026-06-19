"""Reusable decorators for cross-cutting concerns: timing and retry with backoff."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from app.core.logging import get_logger

_logger = get_logger("app.core.decorators")

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

# Exceptions that signal a transient failure worth retrying.
_RETRYABLE_ERRORS = (TimeoutError, ConnectionError, OSError)


def timer(func: F) -> F:
    """Log the wall-clock duration of any async function call."""

    @functools.wraps(func)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        start_ms = time.monotonic() * 1000
        result = await func(*args, **kwargs)
        elapsed_ms = int(time.monotonic() * 1000 - start_ms)
        _logger.debug(
            "[app/core/decorators.py::timer] call finished",
            function=func.__qualname__,
            elapsed_ms=elapsed_ms,
        )
        return result

    return _wrapper  # type: ignore[return-value]


def retry(max_attempts: int, base_delay_seconds: float) -> Callable[[F], F]:
    """Retry an async function with exponential backoff on transient errors.

    Action: Wraps an async function and retries it up to max_attempts times when a
            retryable exception is raised, waiting base_delay_seconds * 2^attempt between tries.
    Trigger: Applied as a decorator on provider generate() methods in app/providers/.
    Arguments:
        max_attempts: Total number of tries (including the first).
        base_delay_seconds: Seconds to wait before the first retry; doubles each round.
    Output: The decorated async function with retry behaviour added.
    """

    def _decorator(func: F) -> F:
        @functools.wraps(func)
        async def _wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt_number in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except _RETRYABLE_ERRORS as exc:
                    last_error = exc
                    if attempt_number == max_attempts - 1:
                        break
                    delay_seconds = base_delay_seconds * (2**attempt_number)
                    _logger.warning(
                        "[app/core/decorators.py::retry] retrying after error",
                        function=func.__qualname__,
                        attempt_number=attempt_number + 1,
                        max_attempts=max_attempts,
                        delay_seconds=delay_seconds,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay_seconds)
            raise RuntimeError(
                f"{func.__qualname__} failed after {max_attempts} attempts"
            ) from last_error

        return _wrapper  # type: ignore[return-value]

    return _decorator
