"""Health check endpoints."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_app_settings
from app.core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {"status": "healthy"}


@router.get("/health/detailed")
async def detailed_health_check(
    settings: Settings = Depends(get_app_settings),
) -> dict:
    """Detailed health check with configuration info."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
        "models": {
            "table_detection": settings.table_detection_model,
            "table_structure": settings.table_structure_model,
        },
        "device": settings.device,
    }

