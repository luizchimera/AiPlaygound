"""API dependencies for dependency injection."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.services.pipeline import DocumentPipeline, PipelineConfig


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async for session in get_db():
        yield session


def get_app_settings() -> Settings:
    """Dependency to get application settings."""
    return get_settings()


def get_pipeline() -> DocumentPipeline:
    """Dependency to get document processing pipeline."""
    config = PipelineConfig()
    return DocumentPipeline(config=config)

