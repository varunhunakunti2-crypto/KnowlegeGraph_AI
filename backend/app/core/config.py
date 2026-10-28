"""Configuration and environment variable validation for KnowledgeGraph AI.
All inference must run locally. No secrets are logged.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


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
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=50)
    
    # Retrieval
    TOP_K_CHUNKS: int = Field(default=10)
    TOP_K_ENTITIES: int = Field(default=10)
    GRAPH_TRAVERSAL_DEPTH: int = Field(default=2)
    MAX_CONTEXT_TOKENS: int = Field(default=4096)
    
    # Ingestion
    UPLOAD_DIR: str = Field(default="./storage/uploads")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50)
    ALLOWED_EXTENSIONS: str = Field(default=".pdf")
    
    # Resource Limits
    MAX_CONCURRENT_REQUESTS: int = Field(default=2)
    MODEL_CONTEXT_LENGTH: int = Field(default=4096)
    BATCH_SIZE: int = Field(default=16)
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]


# Global settings instance
settings = Settings()
