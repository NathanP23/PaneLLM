"""Tests for OpenAI, Anthropic, and Gemini adapters using respx to mock HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.providers.anthropic import AnthropicProvider
from app.providers.anthropic import _build_body as anthropic_body
from app.providers.base import LLMRequest, Message
from app.providers.gemini import GeminiProvider
from app.providers.gemini import _build_body as gemini_body
from app.providers.openai import OpenAIProvider
from app.providers.openai import _build_body as openai_body
from app.providers.registry import build_registry, get_provider


def _req(**kwargs) -> LLMRequest:  # type: ignore[no-untyped-def]
    defaults = dict(model="m", messages=[Message(role="user", content="hi")])
    return LLMRequest(**{**defaults, **kwargs})


# ===========================================================================
# OpenAI
# ===========================================================================

class TestOpenAIBody:
    def test_includes_model_and_messages(self) -> None:
        body = openai_body(_req(model="gpt-4o"))
        assert body["model"] == "gpt-4o"
        assert body["messages"][-1]["content"] == "hi"

    def test_system_prompt_prepended_as_system_message(self) -> None:
        body = openai_body(_req(system="Be terse."))
        assert body["messages"][0] == {"role": "system", "content": "Be terse."}

    def test_json_mode_sets_response_format(self) -> None:
        body = openai_body(_req(response_format="json"))
        assert body["response_format"] == {"type": "json_object"}

    def test_optional_params_omitted_when_none(self) -> None:
        body = openai_body(_req())
        assert "temperature" not in body
        assert "top_p" not in body
        assert "seed" not in body

    def test_extra_params_merged(self) -> None:
        body = openai_body(_req(extra_params={"logprobs": True}))
        assert body["logprobs"] is True


@respx.mock
async def test_openai_adapter_parses_response() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "model": "gpt-4o",
            "choices": [{"message": {"content": "No."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
    )
    response = await OpenAIProvider(api_key="k").generate(_req(model="gpt-4o"))
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
    with pytest.raises(RuntimeError):
        await OpenAIProvider(api_key="k").generate(_req())


# ===========================================================================
# Anthropic
# ===========================================================================

class TestAnthropicBody:
    def test_system_is_top_level_field(self) -> None:
        body = anthropic_body(_req(system="Be concise."))
        assert body["system"] == "Be concise."
        # system must NOT appear as a message
        assert all(m["role"] != "system" for m in body["messages"])

    def test_default_max_tokens_applied_when_none(self) -> None:
        body = anthropic_body(_req())
        assert body["max_tokens"] == 4096

    def test_explicit_max_tokens_respected(self) -> None:
        body = anthropic_body(_req(max_tokens=100))
        assert body["max_tokens"] == 100

    def test_stop_maps_to_stop_sequences(self) -> None:
        body = anthropic_body(_req(stop=["END"]))
        assert body["stop_sequences"] == ["END"]

    def test_optional_params_omitted_when_none(self) -> None:
        body = anthropic_body(_req())
        assert "temperature" not in body
        assert "top_p" not in body


@respx.mock
async def test_anthropic_adapter_parses_response() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Debatable."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 3},
        })
    )
    response = await AnthropicProvider(api_key="k").generate(
        _req(model="claude-3-5-sonnet-20241022")
    )
    assert response.content == "Debatable."
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 3
    assert response.finish_reason == "end_turn"


@respx.mock
async def test_anthropic_adapter_raises_on_error() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(500, json={"error": "server error"})
    )
    with pytest.raises(RuntimeError):
        await AnthropicProvider(api_key="k").generate(_req())


# ===========================================================================
# Gemini
# ===========================================================================

class TestGeminiBody:
    def test_user_message_maps_to_user_role(self) -> None:
        body = gemini_body(_req())
        assert body["contents"][-1]["role"] == "user"

    def test_assistant_role_maps_to_model(self) -> None:
        req = _req(messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ])
        body = gemini_body(req)
        assert body["contents"][-1]["role"] == "model"

    def test_system_prepended_as_user_message(self) -> None:
        body = gemini_body(_req(system="Be brief."))
        assert "[System]" in body["contents"][0]["parts"][0]["text"]

    def test_json_mode_sets_response_mime(self) -> None:
        body = gemini_body(_req(response_format="json"))
        assert body["generationConfig"]["responseMimeType"] == "application/json"

    def test_max_tokens_maps_to_max_output_tokens(self) -> None:
        body = gemini_body(_req(max_tokens=200))
        assert body["generationConfig"]["maxOutputTokens"] == 200

    def test_no_generation_config_when_no_params(self) -> None:
        body = gemini_body(_req())
        assert "generationConfig" not in body


@respx.mock
async def test_gemini_adapter_parses_response() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    ).mock(return_value=httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": "It depends."}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4},
    }))
    response = await GeminiProvider(api_key="k").generate(_req(model="gemini-1.5-pro"))
    assert response.content == "It depends."
    assert response.prompt_tokens == 8
    assert response.completion_tokens == 4
    assert response.finish_reason == "STOP"


@respx.mock
async def test_gemini_adapter_raises_on_error() -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    ).mock(return_value=httpx.Response(403, json={"error": "forbidden"}))
    with pytest.raises(RuntimeError):
        await GeminiProvider(api_key="k").generate(_req(model="gemini-1.5-pro"))


# ===========================================================================
# Registry
# ===========================================================================

def test_registry_always_has_mock() -> None:
    registry = build_registry(Settings(api_key="x"))
    assert "mock" in registry


def test_registry_registers_providers_when_keys_set() -> None:
    registry = build_registry(Settings(
        api_key="x", openai_api_key="o", anthropic_api_key="a", gemini_api_key="g"
    ))
    assert "openai" in registry
    assert "anthropic" in registry
    assert "gemini" in registry


def test_registry_skips_provider_when_key_missing() -> None:
    registry = build_registry(Settings(api_key="x", openai_api_key=""))
    assert "openai" not in registry


def test_get_provider_resolves_vendor_prefix() -> None:
    registry = build_registry(Settings(api_key="x", openai_api_key="o"))
    assert get_provider(registry, "openai/gpt-4o").name == "openai"


def test_get_provider_raises_for_unknown_vendor() -> None:
    registry = build_registry(Settings(api_key="x"))
    with pytest.raises(ValueError, match="No provider"):
        get_provider(registry, "unknown/model-x")
