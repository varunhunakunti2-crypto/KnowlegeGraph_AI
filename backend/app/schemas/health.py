"""Health check schemas."""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class ServiceStatus(BaseModel):
    """Status of a single service."""
    status: str = Field(..., description="Service status")
    error: Optional[str] = Field(default=None)
    models_found: Optional[Dict[str, bool]] = Field(default=None)
    pgvector: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(...)
    services: Dict[str, Any] = Field(...)


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    status: str = Field(...)
    ready: bool = Field(...)
    dependencies: Dict[str, Any] = Field(...)
