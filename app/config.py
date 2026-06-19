"""Single source of configuration, loaded from environment / .env and validated at startup."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable values live here. Read everywhere via get_settings()."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    api_key: str = "dev-local-key"
    cors_origins: str = "http://localhost:3000"

    # Infra (used from M3/M4 onward).
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/debate"
    redis_url: str = "redis://localhost:6379/0"

    # Provider keys (wired in M2; blank => mock provider).
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so config is parsed once per process."""
    return Settings()
