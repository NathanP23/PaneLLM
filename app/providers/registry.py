"""Provider registry: maps 'vendor/model' strings to configured provider instances."""

from __future__ import annotations

from app.config import Settings
from app.core.logging import get_logger
from app.providers.anthropic import AnthropicProvider
from app.providers.base import LLMProvider
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider
from app.providers.openai import OpenAIProvider

_logger = get_logger("app.providers.registry")


def build_registry(settings: Settings) -> dict[str, LLMProvider]:
    """
    Action: Instantiate one provider per configured API key and return a name->provider map.
    Trigger: Called once at worker startup (M4) and in tests via fixture.
    Arguments:
        settings: The validated Settings singleton from app/config.py.
    Output: Dict mapping 'vendor/model' prefix strings to live provider instances.
    """
    registry: dict[str, LLMProvider] = {}

    # Mock is always available — no key required.
    registry["mock"] = MockProvider()

    if settings.openai_api_key:
        registry["openai"] = OpenAIProvider(settings.openai_api_key)
        _logger.info("[app/providers/registry.py::build_registry] registered openai")
    else:
        _logger.warning(
            "[app/providers/registry.py::build_registry] OPENAI_API_KEY not set; skipping"
        )

    if settings.anthropic_api_key:
        registry["anthropic"] = AnthropicProvider(settings.anthropic_api_key)
        _logger.info("[app/providers/registry.py::build_registry] registered anthropic")
    else:
        _logger.warning(
            "[app/providers/registry.py::build_registry] ANTHROPIC_API_KEY not set; skipping"
        )

    if settings.gemini_api_key:
        registry["gemini"] = GeminiProvider(settings.gemini_api_key)
        _logger.info("[app/providers/registry.py::build_registry] registered gemini")
    else:
        _logger.warning(
            "[app/providers/registry.py::build_registry] GEMINI_API_KEY not set; skipping"
        )

    return registry


def get_provider(registry: dict[str, LLMProvider], participant: str) -> LLMProvider:
    """
    Action: Return the provider for a 'vendor/model' string like 'openai/gpt-4o'.
    Trigger: Called by the debate engine per participant turn (M4).
    Arguments:
        registry: The dict returned by build_registry.
        participant: 'vendor/model' string; the vendor prefix is used for lookup.
    Output: The matching LLMProvider instance.
    """
    vendor = participant.split("/")[0]
    provider = registry.get(vendor)
    if provider is None:
        raise ValueError(f"No provider registered for vendor '{vendor}' (from '{participant}')")
    return provider
