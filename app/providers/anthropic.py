"""Anthropic provider adapter. Translates LLMRequest <-> Anthropic messages API."""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.decorators import retry, timer
from app.core.logging import get_logger
from app.providers.base import LLMProvider, LLMRequest, LLMResponse

_logger = get_logger("app.providers.anthropic")
_BASE_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

# Anthropic requires max_tokens; use a safe default when caller omits it.
_DEFAULT_MAX_TOKENS = 4096


def _build_body(request: LLMRequest) -> dict[str, Any]:
    """Build Anthropic messages API body; system prompt is a top-level field."""
    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }
    if request.system:
        body["system"] = request.system
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop:
        body["stop_sequences"] = request.stop
    if request.tools:
        body["tools"] = request.tools
    if request.tool_choice:
        body["tool_choice"] = request.tool_choice
    body.update(request.extra_params)
    return body


def _parse_response(raw: dict[str, Any], start_ms: float, request_model: str) -> LLMResponse:
    """Extract our normalized shape from the raw Anthropic response dict."""
    content_blocks: list[dict[str, Any]] = raw.get("content", [])
    text = " ".join(block["text"] for block in content_blocks if block.get("type") == "text")
    usage = raw.get("usage", {})
    return LLMResponse(
        content=text,
        model=raw.get("model", request_model),
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        finish_reason=raw.get("stop_reason"),
        latency_ms=int(time.monotonic() * 1000 - start_ms),
        raw=raw,
    )


class AnthropicProvider(LLMProvider):
    """Calls Anthropic messages endpoint via httpx."""

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        """
        Action: Store auth credentials for all subsequent requests.
        Trigger: Called by the provider registry when the app starts.
        Arguments:
            api_key: Anthropic secret key from environment (never hardcoded).
        Output: Configured provider instance ready to generate.
        """
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    @timer
    @retry(max_attempts=3, base_delay_seconds=1.0)
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Action: Call Anthropic messages API and return a normalized LLMResponse.
        Trigger: Called by the debate engine for each participant turn.
        Arguments:
            request: Provider-agnostic request; unsupported fields are silently dropped.
        Output: Normalized LLMResponse with content, token counts, latency.
        """
        start_ms = time.monotonic() * 1000
        body = _build_body(request)
        _logger.debug(
            "[app/providers/anthropic.py::generate] sending request",
            model=request.model,
        )
        async with httpx.AsyncClient(timeout=request.timeout_seconds) as client:
            response = await client.post(_BASE_URL, json=body, headers=self._headers)

        if response.status_code != 200:
            _logger.error(
                "[app/providers/anthropic.py::generate] API error",
                status_code=response.status_code,
                body=response.text[:200],
            )
            raise ConnectionError(f"Anthropic API returned {response.status_code}")

        raw: dict[str, Any] = response.json()
        return _parse_response(raw, start_ms, request.model)
