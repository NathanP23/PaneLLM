"""Provider-agnostic LLM interface. Adapters map these shapes to/from a vendor API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMRequest(BaseModel):
    """Everything the engine knows how to ask for. Adapters drop unsupported fields."""

    model: str
    messages: list[Message]
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    seed: int | None = None
    response_format: str | None = None  # e.g. "json"
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    stream: bool = False
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict)  # provider-specific passthrough


class LLMResponse(BaseModel):
    """Normalized result. `raw` keeps the untouched vendor payload for debugging."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    latency_ms: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMProvider(ABC):
    """Implement one per vendor. The engine only ever sees this interface."""

    name: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one completion. Must raise on unrecoverable failure."""
        raise NotImplementedError
