"""KnowledgeGraph AI - FastAPI Backend
Fully local GraphRAG document assistant. All inference runs locally.
"""
import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core import memory
from app.core.config import settings
from app.core.logging import logger
from app.core.security import is_loopback_host
from app.api.v1 import api_router
from app.db import db_manager
from app.db.documents import DocumentDB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting KnowledgeGraph AI API")
    if not is_loopback_host(settings.BACKEND_HOST):
        logger.warning(
            "BACKEND_HOST is not a loopback address. The API is reachable from "
            "the local network. Keep Neo4j/PostgreSQL bound to localhost and "
            "only do this behind a trusted firewall."
        )
    logger.info(
        f"Memory profile={settings.MEMORY_PROFILE}; "
        f"concurrency={settings.MAX_CONCURRENT_REQUESTS}; "
        f"index_batch={settings.INDEX_BATCH_SIZE}; "
        f"context={settings.MODEL_CONTEXT_LENGTH}"
    )
    await db_manager.startup()
    try:
        DocumentDB.init_table()
    except Exception as exc:
        logger.warning(f"Document table initialization skipped: {type(exc).__name__}")

    # Release the local models after an idle period. The loop is a no-op when
    # OLLAMA_IDLE_UNLOAD_SECONDS=0 or the memory guard is disabled.
    idle_monitor = None
    if settings.MEMORY_GUARD_ENABLED and settings.OLLAMA_IDLE_UNLOAD_SECONDS > 0:
        idle_monitor = asyncio.create_task(memory.idle_unload_monitor())

    try:
        yield
    finally:
        if idle_monitor is not None:
            idle_monitor.cancel()
            with suppress(asyncio.CancelledError):
                await idle_monitor
        # Leave the GPU free when the backend stops.
        if settings.MEMORY_GUARD_ENABLED:
            try:
                await asyncio.to_thread(memory.unload_app_models)
            except Exception as exc:
                logger.debug(f"Shutdown model unload skipped: {type(exc).__name__}")
        logger.info("Shutting down KnowledgeGraph AI API")
        await db_manager.shutdown()


app = FastAPI(
    title="KnowledgeGraph AI",
    description="Fully local GraphRAG document assistant",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS - restricted to configured local frontend origins. Wildcards
# are rejected by Settings so credentials can never be combined with '*'.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
    max_age=600,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add conservative headers without changing API behavior."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(memory.InsufficientMemoryError)
async def insufficient_memory_handler(request: Request, exc: memory.InsufficientMemoryError):
    """Hardware limits are actionable, so return a clear 503 instead of a 500."""
    logger.error("Request stopped: insufficient local memory")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": str(exc),
            "status_code": 503,
        },
    )


# Consistent error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Never include the exception message: it may contain SQL, paths, or
    # connection strings. Type-only logging is enough for local debugging.
    logger.error(f"Unhandled exception: {type(exc).__name__}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "KnowledgeGraph AI API",
        "version": "0.1.0",
        "local_only": True,
    }


# Include versioned API routes
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.BACKEND_HOST, port=settings.BACKEND_PORT)
