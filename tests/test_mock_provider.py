from __future__ import annotations

from app.providers.base import LLMRequest, Message
from app.providers.mock import MockProvider


async def test_mock_provider_echoes_last_user_message() -> None:
    provider = MockProvider()
    request = LLMRequest(
        model="m1",
        messages=[Message(role="user", content="hello world")],
    )
    response = await provider.generate(request)
    assert response.content == "[mock:m1] hello world"
    assert response.completion_tokens > 0
