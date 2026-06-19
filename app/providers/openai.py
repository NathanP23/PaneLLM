"""OpenAI provider adapter. Translates LLMRequest <-> OpenAI chat-completions API."""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.decorators import retry, timer
from app.core.logging import get_logger
from app.providers.base import LLMProvider, LLMRequest, LLMResponse

_logger = get_logger("app.providers.openai")
_BASE_URL = "https://api.openai.com/v1/chat/completions"


def _build_messages(request: LLMRequest) -> list[dict[str, str]]:
    """Prepend the system prompt as a system message if present."""
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.extend({"role": m.role, "content": m.content} for m in request.messages)
    return messages


def _build_body(request: LLMRequest) -> dict[str, Any]:
    """Build the JSON body for the OpenAI API, dropping unsupported/unset params."""
    body: dict[str, Any] = {
        "model": request.model,
        "messages": _build_messages(request),
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.stop:
        body["stop"] = request.stop
    if request.seed is not None:
        body["seed"] = request.seed
    if request.response_format == "json":
        body["response_format"] = {"type": "json_object"}
    if request.tools:
        body["tools"] = request.tools
    if request.tool_choice:
        body["tool_choice"] = request.tool_choice
    body.update(request.extra_params)
    return body


def _parse_response(raw: dict[str, Any], start_ms: float, request_model: str) -> LLMResponse:
    """Extract our normalized shape from the raw OpenAI response dict."""
    choice = raw["choices"][0]
    usage = raw.get("usage", {})
    return LLMResponse(
        content=choice["message"]["content"] or "",
        model=raw.get("model", request_model),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        finish_reason=choice.get("finish_reason"),
        latency_ms=int(time.monotonic() * 1000 - start_ms),
        raw=raw,
    )


class OpenAIProvider(LLMProvider):
    """Calls OpenAI chat-completions endpoint via httpx."""

    name = "openai"

    def __init__(self, api_key: str) -> None:
        """
        Action: Store auth credentials for all subsequent requests.
        Trigger: Called by the provider registry when the app starts.
        Arguments:
            api_key: OpenAI secret key from environment (never hardcoded).
        Output: Configured provider instance ready to generate.
        """
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @timer
    @retry(max_attempts=3, base_delay_seconds=1.0)
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Action: Call OpenAI chat-completions and return a normalized LLMResponse.
        Trigger: Called by the debate engine for each participant turn.
        Arguments:
            request: Provider-agnostic request; unsupported fields are silently dropped.
        Output: Normalized LLMResponse with content, token counts, latency.
        """
        start_ms = time.monotonic() * 1000
        body = _build_body(request)
        _logger.debug(
            "[app/providers/openai.py::generate] sending request",
            model=request.model,
        )
        async with httpx.AsyncClient(timeout=request.timeout_seconds) as client:
            response = await client.post(_BASE_URL, json=body, headers=self._headers)

        if response.status_code != 200:
            _logger.error(
                "[app/providers/openai.py::generate] API error",
                status_code=response.status_code,
                body=response.text[:200],
            )
            raise ConnectionError(f"OpenAI API returned {response.status_code}")

        raw: dict[str, Any] = response.json()
        return _parse_response(raw, start_ms, request.model)
