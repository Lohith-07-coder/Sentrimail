"""Centralized, typed application configuration."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from the environment and an optional local ``.env`` file."""

    app_name: str = "SentriMail"
    debug: bool = False

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "sentrimail"

    secret_key: str | None = None
    jwt_secret_key: str | None = None
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1_440
    refresh_token_expire_days: int = 7

    session_cookie_name: str = "sentrimail_session"
    refresh_cookie_name: str = "sentrimail_refresh"
    secure_cookies: bool = False
    cookie_samesite: str = "lax"

    mail_server: str | None = None
    mail_port: int = 587
    mail_use_tls: bool = True
    mail_username: str | None = None
    mail_password: str | None = None
    mail_default_sender: str = "noreply@sentrimail.com"

    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: object) -> bool:
        """Default unexpected inherited DEBUG values to the secure setting."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""
    return Settings()
