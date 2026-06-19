"""Deterministic mock provider: no API key, no network. Used in dev and CI."""

from __future__ import annotations

from app.providers.base import LLMProvider, LLMRequest, LLMResponse


class MockProvider(LLMProvider):
    name = "mock"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Echo the last user message so tests can assert on deterministic output."""
        last_user = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        )
        content = f"[mock:{request.model}] {last_user}"
        return LLMResponse(
            content=content,
            model=request.model,
            prompt_tokens=len(last_user.split()),
            completion_tokens=len(content.split()),
            finish_reason="stop",
        )
