"""Tests for OpenAI, Anthropic, and Gemini adapters using respx to mock HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.providers.anthropic import AnthropicProvider
from app.providers.base import LLMRequest, Message
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider
from app.providers.registry import build_registry, get_provider

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_request(model: str) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=[Message(role="user", content="Is a hot dog a sandwich?")],
        system="You are a helpful assistant.",
        max_tokens=100,
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

@respx.mock
async def test_openai_adapter_parses_response() -> None:
    fake_response = {
        "id": "chatcmpl-1",
        "model": "gpt-4o",
        "choices": [{"message": {"content": "No."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=fake_response)
    )
    provider = OpenAIProvider(api_key="test-key")
    response = await provider.generate(_make_request("gpt-4o"))

    assert response.content == "No."
    assert response.model == "gpt-4o"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.finish_reason == "stop"


@respx.mock
async def test_openai_adapter_raises_on_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "rate limit"})
    )
    provider = OpenAIProvider(api_key="test-key")
    with pytest.raises(RuntimeError):
        await provider.generate(_make_request("gpt-4o"))


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

@respx.mock
async def test_anthropic_adapter_parses_response() -> None:
    fake_response = {
        "id": "msg-1",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "Debatable."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=fake_response)
    )
    provider = AnthropicProvider(api_key="test-key")
    response = await provider.generate(_make_request("claude-3-5-sonnet-20241022"))

    assert response.content == "Debatable."
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 3
    assert response.finish_reason == "end_turn"


@respx.mock
async def test_anthropic_adapter_raises_on_error() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(500, json={"error": "server error"})
    )
    provider = AnthropicProvider(api_key="test-key")
    with pytest.raises(RuntimeError):
        await provider.generate(_make_request("claude-3-5-sonnet-20241022"))


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

@respx.mock
async def test_gemini_adapter_parses_response() -> None:
    fake_response = {
        "candidates": [
            {
                "content": {"parts": [{"text": "It depends."}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4},
    }
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    ).mock(return_value=httpx.Response(200, json=fake_response))

    provider = GeminiProvider(api_key="test-key")
    response = await provider.generate(_make_request("gemini-1.5-pro"))

    assert response.content == "It depends."
    assert response.prompt_tokens == 8
    assert response.completion_tokens == 4
    assert response.finish_reason == "STOP"


@respx.mock
async def test_gemini_adapter_raises_on_error() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    ).mock(return_value=httpx.Response(403, json={"error": "forbidden"}))

    provider = GeminiProvider(api_key="test-key")
    with pytest.raises(RuntimeError):
        await provider.generate(_make_request("gemini-1.5-pro"))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_always_has_mock() -> None:
    settings = Settings(
        api_key="x",
        anthropic_api_key="",
        openai_api_key="",
        gemini_api_key="",
    )
    registry = build_registry(settings)
    assert "mock" in registry


def test_registry_registers_providers_when_keys_set() -> None:
    settings = Settings(
        api_key="x",
        anthropic_api_key="a-key",
        openai_api_key="o-key",
        gemini_api_key="g-key",
    )
    registry = build_registry(settings)
    assert "openai" in registry
    assert "anthropic" in registry
    assert "gemini" in registry


def test_get_provider_resolves_vendor_prefix() -> None:
    settings = Settings(api_key="x", openai_api_key="o-key")
    registry = build_registry(settings)
    provider = get_provider(registry, "openai/gpt-4o")
    assert provider.name == "openai"


def test_get_provider_raises_for_unknown_vendor() -> None:
    registry = build_registry(Settings(api_key="x"))
    with pytest.raises(ValueError, match="No provider"):
        get_provider(registry, "unknown/model-x")
