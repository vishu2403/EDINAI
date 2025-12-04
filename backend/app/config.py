"""Centralized configuration for the modular backend example."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any

from pydantic import Field, field_validator, PostgresDsn, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App Settings
    app_name: str = Field("EDINAI Modular Backend", env="APP_NAME")
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("production", env="ENVIRONMENT")
    
    # Database Configuration
    database_url: PostgresDsn = Field(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/inai", 
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(10, env="DATABASE_MAX_OVERFLOW")
    database_pool_recycle: int = Field(3600, env="DATABASE_POOL_RECYCLE")
    database_pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")
    
    # Security
    secret_key: str = Field("your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    access_token_expire_days: int = Field(7, env="ACCESS_TOKEN_EXPIRE_DAYS")
    
    # CORS
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://127.0.0.1:3000"], 
        env="CORS_ORIGINS"
    )
    
    # API Settings
    api_v1_prefix: str = "/api/v1"
    
    # File Uploads
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = ["image/jpeg", "image/png", "application/pdf"]
    
    # Model Config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    @validator("database_url", pre=True)
    def assemble_db_connection(cls, v: str | PostgresDsn) -> str | PostgresDsn:
        if isinstance(v, str) and v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+psycopg2://", 1)
        return v
    
    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
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
