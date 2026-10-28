"""Model status endpoint."""
from fastapi import APIRouter
from app.core.config import settings
from app.services.llm import llm_service
from app.services.embeddings import embedding_service
from typing import Dict, Any

router = APIRouter()


@router.get("/status")
async def model_status() -> Dict[str, Any]:
    """Get local model status without exposing sensitive info."""
    llm_status = await llm_service.check_model_availability()
    embed_status = await embedding_service.check_model_availability()
    
    return {
        # Only report the host; credentials are never part of this config.
        "ollama_host": settings.OLLAMA_HOST,
        "llm": llm_status,
        "embeddings": embed_status,
        "all_available": llm_status["available"] and embed_status["available"]
    }
