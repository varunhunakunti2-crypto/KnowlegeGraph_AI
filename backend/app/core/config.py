"""Configuration and environment variable validation for KnowledgeGraph AI.
All inference must run locally. No secrets are logged.
"""
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import List

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Load an optional backend/.env first, then the repo-root .env. Absolute paths
# keep configuration working regardless of the process working directory (the
# README starts uvicorn from backend/). Later files take precedence.
_ENV_FILES = (_BACKEND_DIR / ".env", _REPO_ROOT / ".env")

# Applied by MEMORY_PROFILE=low-memory unless the field is explicitly set.
# These values keep peak VRAM/RAM inside an RTX 4050 (6 GB) + 16 GB laptop.
_LOW_MEMORY_DEFAULTS = {
    "MODEL_CONTEXT_LENGTH": 2048,
    "OLLAMA_MAX_TOKENS": 384,
    "OLLAMA_NUM_BATCH": 128,
    "OLLAMA_KEEP_ALIVE": "2m",
    "OLLAMA_IDLE_UNLOAD_SECONDS": 120,
    "OLLAMA_SINGLE_MODEL_RESIDENT": True,
    "MAX_CONTEXT_TOKENS": 1536,
    "EMBEDDING_BATCH_SIZE": 8,
    "BATCH_SIZE": 8,
    "INDEX_BATCH_SIZE": 4,
    "MAX_INDEX_BATCH_FAILURES": 2,
    "MAX_CONCURRENT_REQUESTS": 1,
    "MAX_CONCURRENT_UPLOADS": 1,
    "TOP_K_CHUNKS": 5,
    "MAX_EVIDENCE_ITEMS": 8,
    "GRAPH_TRAVERSAL_DEPTH": 1,
    "MAX_GRAPH_EXTRACTION_CHUNKS": 25,
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    ENV: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    
    # Backend
    BACKEND_HOST: str = Field(default="127.0.0.1")
    BACKEND_PORT: int = Field(default=8000)
    BACKEND_CORS_ORIGINS: str = Field(default="http://localhost:3000,http://127.0.0.1:3000")
    
    # Frontend
    NEXT_PUBLIC_API_URL: str = Field(default="http://localhost:8000")
    
    # Ollama
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="qwen3:4b")
    OLLAMA_EMBED_MODEL: str = Field(default="nomic-embed-text")
    OLLAMA_TIMEOUT: int = Field(default=300)
    OLLAMA_TEMPERATURE: float = Field(default=0.1)
    OLLAMA_MAX_TOKENS: int = Field(default=2048)
    # How long Ollama keeps a model resident after a request (for example
    # "5m", "30s", "1h", or -1 to keep it loaded indefinitely).
    OLLAMA_KEEP_ALIVE: str = Field(default="5m")
    # Unload the app's models after this many seconds without model work.
    # 0 disables idle unloading. Low-memory profile defaults to 120s.
    OLLAMA_IDLE_UNLOAD_SECONDS: int = Field(default=900, ge=0, le=86400)
    # Prompt-processing batch size. Smaller values reduce peak VRAM.
    OLLAMA_NUM_BATCH: int = Field(default=512, ge=1, le=2048)
    # In low-memory mode only one model may stay resident at a time.
    OLLAMA_SINGLE_MODEL_RESIDENT: bool = Field(default=False)
    
    # Neo4j
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="changeme")
    NEO4J_DATABASE: str = Field(default="neo4j")
    
    # PostgreSQL
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="knowledgegraph")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="changeme")
    DATABASE_URL: str = Field(default="postgresql+psycopg://postgres:changeme@localhost:5432/knowledgegraph")
    
    # Embedding
    EMBEDDING_DIMENSION: int = Field(default=768)
    EMBEDDING_BATCH_SIZE: int = Field(default=32)
    
    # Chunking
    CHUNK_SIZE: int = Field(default=512, ge=64, le=8192)
    CHUNK_OVERLAP: int = Field(default=50, ge=0, le=4096)
    
    # Retrieval
    TOP_K_CHUNKS: int = Field(default=10)
    TOP_K_ENTITIES: int = Field(default=10)
    GRAPH_TRAVERSAL_DEPTH: int = Field(default=2)
    MAX_CONTEXT_TOKENS: int = Field(default=4096)
    
    # Ingestion
    UPLOAD_DIR: str = Field(default="./storage/uploads")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, ge=1, le=500)
    ALLOWED_EXTENSIONS: str = Field(default=".pdf")
    MAX_PDF_PAGES: int = Field(default=500, ge=1, le=5000)
    MAX_EXTRACTED_TEXT_CHARS: int = Field(default=5_000_000, ge=10_000)
    MAX_CHUNKS_PER_DOCUMENT: int = Field(default=10_000, ge=1, le=100_000)
    ALLOW_PDF_ACTIVE_CONTENT: bool = Field(default=False)
    UPLOAD_PROCESS_TIMEOUT_SECONDS: int = Field(default=180, ge=10, le=1800)
    
    # Resource Limits
    MAX_CONCURRENT_REQUESTS: int = Field(default=2, ge=1, le=8)
    MAX_CONCURRENT_UPLOADS: int = Field(default=2, ge=1, le=8)
    MODEL_CONTEXT_LENGTH: int = Field(default=4096, ge=512, le=32768)
    BATCH_SIZE: int = Field(default=16, ge=1, le=128)
    # Rows committed per indexing batch. Small batches make indexing resumable
    # after a crash or OOM without losing completed work.
    INDEX_BATCH_SIZE: int = Field(default=8, ge=1, le=128)
    MAX_INDEX_BATCH_FAILURES: int = Field(default=3, ge=1, le=20)
    DB_CONNECT_TIMEOUT_SECONDS: int = Field(default=5, ge=1, le=60)
    NEO4J_QUERY_TIMEOUT_SECONDS: int = Field(default=10, ge=1, le=120)
    OLLAMA_MAX_RETRIES: int = Field(default=2, ge=0, le=5)

    # Memory profile and guard. "low-memory" caps context, batches, concurrency
    # and graph work for the 6 GB / 16 GB laptop target. Explicit environment
    # values always win over profile defaults.
    MEMORY_PROFILE: str = Field(default="default")
    MEMORY_GUARD_ENABLED: bool = Field(default=True)
    OLLAMA_UNLOAD_ON_MEMORY_PRESSURE: bool = Field(default=True)
    MIN_FREE_MEMORY_MB: int = Field(default=1024, ge=128, le=32768)
    MIN_FREE_VRAM_MB: int = Field(default=512, ge=0, le=24576)
    MEMORY_CHECK_INTERVAL_SECONDS: int = Field(default=30, ge=5, le=600)
    
    # Retrieval / graph safety
    MAX_GRAPH_LIMIT: int = Field(default=200, ge=1, le=1000)
    MAX_GRAPH_TRAVERSAL_DEPTH: int = Field(default=3, ge=1, le=3)
    MAX_GRAPH_EXTRACTION_CHUNKS: int = Field(default=100, ge=1, le=1000)
    MAX_EVIDENCE_ITEMS: int = Field(default=20, ge=1, le=100)
    MAX_EVIDENCE_CHARS: int = Field(default=4000, ge=100, le=50000)
    MAX_QUESTION_CHARS: int = Field(default=2000, ge=1, le=10000)
    MAX_TOP_K: int = Field(default=50, ge=1, le=200)
    MAX_EXTRACTED_ENTITIES_PER_CHUNK: int = Field(default=30, ge=1, le=200)
    MAX_EXTRACTED_RELATIONSHIPS_PER_CHUNK: int = Field(default=30, ge=1, le=200)
    GRAPH_EXTRACTION_TIMEOUT_SECONDS: int = Field(default=600, ge=30, le=3600)
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    
    class Config:
        env_file = _ENV_FILES
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @model_validator(mode="after")
    def _validate_security_settings(self):
        """Reject configurations that would weaken local-only defaults."""
        if self.MEMORY_PROFILE not in {"default", "low-memory"}:
            raise ValueError(
                "MEMORY_PROFILE must be 'default' or 'low-memory'"
            )
        if self.MEMORY_PROFILE == "low-memory":
            # Explicit environment/constructor values always win. These are
            # only applied when the operator selected the profile without
            # overriding the individual limits.
            for field, value in _LOW_MEMORY_DEFAULTS.items():
                if field not in self.model_fields_set:
                    object.__setattr__(self, field, value)

        origins = [o.strip() for o in self.BACKEND_CORS_ORIGINS.split(",") if o.strip()]
        if "*" in origins:
            raise ValueError(
                "BACKEND_CORS_ORIGINS must list explicit frontend origins; '*' is not allowed"
            )
        for origin in origins:
            if not origin.startswith(("http://", "https://")):
                raise ValueError(f"Invalid CORS origin: {origin}")
        if self.MAX_CONTEXT_TOKENS > self.MODEL_CONTEXT_LENGTH:
            # Keep the retrieval budget inside the model window.
            self.MAX_CONTEXT_TOKENS = self.MODEL_CONTEXT_LENGTH
        if self.GRAPH_TRAVERSAL_DEPTH > self.MAX_GRAPH_TRAVERSAL_DEPTH:
            self.GRAPH_TRAVERSAL_DEPTH = self.MAX_GRAPH_TRAVERSAL_DEPTH
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            # Overlap must be smaller than the window or chunking never advances.
            self.CHUNK_OVERLAP = max(0, self.CHUNK_SIZE // 10)
        return self
    
    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string.

        Wildcards are rejected in validation. Trailing slashes are normalized
        because browsers send origins without them.
        """
        origins = []
        for raw in self.BACKEND_CORS_ORIGINS.split(","):
            origin = raw.strip().rstrip("/")
            if not origin:
                continue
            if origin == "*":
                raise ValueError("Wildcard CORS origins are not allowed")
            if not origin.startswith(("http://", "https://")):
                raise ValueError(f"Invalid CORS origin: {origin}")
            origins.append(origin)
        return origins


# Global settings instance
settings = Settings()
