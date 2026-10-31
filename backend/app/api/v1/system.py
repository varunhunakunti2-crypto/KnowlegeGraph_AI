"""Runtime system status and local model lifecycle endpoints.

Measurements come from the running machine (psutil, nvidia-smi, Ollama
``/api/ps``). They are not estimates and do not assume CUDA is active.
"""
import asyncio

from fastapi import APIRouter, HTTPException, status

from app.core import memory
from app.core.logging import logger

router = APIRouter()


@router.get("/status")
async def system_status():
    """Measured RAM/GPU state, resident models, pressure and active limits."""
    try:
        return await asyncio.to_thread(memory.runtime_status)
    except Exception as exc:
        logger.error(f"System status failed: {type(exc).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System status unavailable",
        )


@router.get("/gpu")
async def gpu_status():
    """A fresh nvidia-smi sample, or ``measured: false`` when no GPU is visible."""
    sample = await asyncio.to_thread(memory.gpu_status, True)
    return {
        "measured": sample is not None,
        "gpu": sample,
        "note": (
            None if sample is not None
            else "nvidia-smi was not found or returned no data; Ollama may still "
                 "use the CPU. Check `ollama ps` to see where the model runs."
        ),
    }


@router.post("/unload")
async def unload_models():
    """Unload this app's Ollama models now (frees VRAM/RAM immediately)."""
    try:
        unloaded = await asyncio.to_thread(memory.unload_app_models)
        return {"unloaded": unloaded, "count": len(unloaded)}
    except Exception as exc:
        logger.error(f"Model unload failed: {type(exc).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not unload models",
        )
