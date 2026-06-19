"""Tests for @timer and @retry decorators."""

from __future__ import annotations

import pytest

from app.core.decorators import retry, timer


async def test_timer_returns_result() -> None:
    @timer
    async def _add(first_value: int, second_value: int) -> int:
        return first_value + second_value

    assert await _add(2, 3) == 5


async def test_retry_succeeds_on_second_attempt() -> None:
    call_count = 0

    @retry(max_attempts=3, base_delay_seconds=0.0)
    async def _flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("temporary failure")
        return "ok"

    result = await _flaky()
    assert result == "ok"
    assert call_count == 2


async def test_retry_raises_after_max_attempts() -> None:
    @retry(max_attempts=2, base_delay_seconds=0.0)
    async def _always_fails() -> None:
        raise ConnectionError("permanent")

    with pytest.raises(RuntimeError, match="2 attempts"):
        await _always_fails()
