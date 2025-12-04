"""Centralized configuration for the modular backend example."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = Field("EDINAI Modular Backend", env="APP_NAME")
    database_url: str = Field(
        "postgresql://postgres:postgres@localhost:5432/inai", env="DATABASE_URL"
    )
    secret_key: str = Field("super-secret-key", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    access_token_expire_days: int = Field(7, env="ACCESS_TOKEN_EXPIRE_DAYS")
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    algorithm: str = Field("HS256", env="ALGORITHM")
    allowed_email_domains_raw: str = Field("gmail.com", env="ALLOWED_EMAIL_DOMAINS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], env="CORS_ORIGINS")
    fernet_key: Optional[str] = Field(None, env="FERNET_KEY")
    default_language: str = Field("English", env="DEFAULT_LANGUAGE")
    default_lecture_duration: int = Field(45, env="DEFAULT_LECTURE_DURATION")
    dev_admin_email: Optional[str] = Field("dev_admin@inai.dev", env="DEV_ADMIN_EMAIL")
    dev_admin_password: Optional[str] = Field("DevAdmin@123", env="DEV_ADMIN_PASSWORD")
    dev_admin_name: str = Field("Dev Admin", env="DEV_ADMIN_NAME")
    dev_admin_package: str = Field("trial", env="DEV_ADMIN_PACKAGE")
    dev_admin_expiry_days: int = Field(365, env="DEV_ADMIN_EXPIRY_DAYS")
    password_reset_url: Optional[str] = Field(None, env="PASSWORD_RESET_URL")

    public_base_url: Optional[str] = Field(
        None,
        env="PUBLIC_BASE_URL",
        description="Base URL (e.g., https://example.com) used when constructing absolute media links",
    )
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @property
    def allowed_email_domains(self) -> List[str]:
        return [
            domain.strip().lower()
            for domain in self.allowed_email_domains_raw.split(",")
            if domain.strip()
        ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: List[str] | str) -> List[str]:
        if isinstance(value, str):
            if value.strip() == "*":
                return ["*"]
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance so values are computed once."""

    return Settings()


settings = get_settings()
