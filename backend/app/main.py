"""KnowledgeGraph AI - FastAPI Backend
Fully local GraphRAG document assistant. All inference runs locally.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logging import logger
from app.api.v1 import api_router
from app.db import db_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting KnowledgeGraph AI API")
    await db_manager.startup()
    yield
    logger.info("Shutting down KnowledgeGraph AI API")
    await db_manager.shutdown()


app = FastAPI(
    title="KnowledgeGraph AI",
    description="Fully local GraphRAG document assistant",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS - restricted to local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Consistent error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
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
