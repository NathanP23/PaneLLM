"""Tests for app/config.py — settings loading and derived properties."""

from __future__ import annotations

from app.config import Settings


def test_default_log_level() -> None:
    settings = Settings(api_key="x")
    assert settings.log_level == "INFO"


def test_default_api_key_used_in_tests() -> None:
    settings = Settings(api_key="dev-local-key")
    assert settings.api_key == "dev-local-key"


def test_cors_origin_list_splits_on_comma() -> None:
    settings = Settings(api_key="x", cors_origins="http://a.com,http://b.com")
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]


def test_cors_origin_list_strips_whitespace() -> None:
    settings = Settings(api_key="x", cors_origins="http://a.com , http://b.com")
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]


def test_cors_origin_list_single_origin() -> None:
    settings = Settings(api_key="x", cors_origins="http://localhost:3000")
    assert settings.cors_origin_list == ["http://localhost:3000"]


def test_empty_provider_keys_by_default() -> None:
    settings = Settings(api_key="x")
    assert settings.openai_api_key == ""
    assert settings.anthropic_api_key == ""
    assert settings.gemini_api_key == ""
