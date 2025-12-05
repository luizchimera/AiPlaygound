"""Application configuration and settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "hgTransformer"
    app_version: str = "0.1.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hgtransformer"

    # File Storage
    upload_dir: Path = Path("./uploads")
    max_file_size_mb: int = 50

    # Model Configuration
    table_detection_model: str = "microsoft/table-transformer-detection"
    table_structure_model: str = "microsoft/table-transformer-structure-recognition"
    device: str = "cpu"  # "cuda" for GPU

    # Processing
    detection_threshold: float = 0.7
    structure_threshold: float = 0.6

    # API
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

