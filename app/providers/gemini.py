"""Gemini provider adapter. Translates LLMRequest <-> Google Gemini generateContent API."""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.decorators import retry, timer
from app.core.logging import get_logger
from app.providers.base import LLMProvider, LLMRequest, LLMResponse

_logger = get_logger("app.providers.gemini")
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _build_contents(request: LLMRequest) -> list[dict[str, Any]]:
    """Convert our messages to Gemini 'contents' format (role + parts)."""
    # Gemini uses 'user' and 'model' (not 'assistant') as role names.
    role_map = {"user": "user", "assistant": "model", "system": "user"}
    contents: list[dict[str, Any]] = []
    if request.system:
        contents.append({"role": "user", "parts": [{"text": f"[System]: {request.system}"}]})
    for message in request.messages:
        contents.append({
            "role": role_map.get(message.role, "user"),
            "parts": [{"text": message.content}],
        })
    return contents


def _build_body(request: LLMRequest) -> dict[str, Any]:
    """Build Gemini generateContent body."""
    body: dict[str, Any] = {"contents": _build_contents(request)}
    generation_config: dict[str, Any] = {}
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.top_p is not None:
        generation_config["topP"] = request.top_p
    if request.max_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_tokens
    if request.stop:
        generation_config["stopSequences"] = request.stop
    if request.response_format == "json":
        generation_config["responseMimeType"] = "application/json"
    if generation_config:
        body["generationConfig"] = generation_config
    body.update(request.extra_params)
    return body


def _parse_response(raw: dict[str, Any], start_ms: float, request_model: str) -> LLMResponse:
    """Extract our normalized shape from the raw Gemini response dict."""
    candidates: list[dict[str, Any]] = raw.get("candidates", [])
    text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = " ".join(part.get("text", "") for part in parts)
    usage = raw.get("usageMetadata", {})
    return LLMResponse(
        content=text,
        model=request_model,
        prompt_tokens=usage.get("promptTokenCount", 0),
        completion_tokens=usage.get("candidatesTokenCount", 0),
        finish_reason=candidates[0].get("finishReason") if candidates else None,
        latency_ms=int(time.monotonic() * 1000 - start_ms),
        raw=raw,
    )


class GeminiProvider(LLMProvider):
    """Calls Gemini generateContent endpoint via httpx."""

    name = "gemini"

    def __init__(self, api_key: str) -> None:
        """
        Action: Store auth credentials for all subsequent requests.
        Trigger: Called by the provider registry when the app starts.
        Arguments:
            api_key: Google AI API key from environment (never hardcoded).
        Output: Configured provider instance ready to generate.
        """
        self._api_key = api_key

    @timer
    @retry(max_attempts=3, base_delay_seconds=1.0)
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Action: Call Gemini generateContent and return a normalized LLMResponse.
        Trigger: Called by the debate engine for each participant turn.
        Arguments:
            request: Provider-agnostic request; unsupported fields are silently dropped.
        Output: Normalized LLMResponse with content, token counts, latency.
        """
        start_ms = time.monotonic() * 1000
        url = _BASE_URL.format(model=request.model)
        body = _build_body(request)
        _logger.debug(
            "[app/providers/gemini.py::generate] sending request",
            model=request.model,
        )
        async with httpx.AsyncClient(timeout=request.timeout_seconds) as client:
            response = await client.post(url, json=body, params={"key": self._api_key})

        if response.status_code != 200:
            _logger.error(
                "[app/providers/gemini.py::generate] API error",
                status_code=response.status_code,
                body=response.text[:200],
            )
            raise ConnectionError(f"Gemini API returned {response.status_code}")

        raw: dict[str, Any] = response.json()
        return _parse_response(raw, start_ms, request.model)
