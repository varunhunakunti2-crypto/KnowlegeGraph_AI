"""Embedding service using Ollama (nomic-embed-text). No paid APIs."""
from ollama import Client
from app.core.config import settings
from app.core.logging import logger
from app.core.resources import model_semaphore
from typing import List, Dict, Any
import asyncio


class EmbeddingService:
    """Service for embeddings via local Ollama."""
    
    def __init__(self):
        self.client = Client(host=settings.OLLAMA_HOST, timeout=settings.OLLAMA_TIMEOUT)
        self.model = settings.OLLAMA_EMBED_MODEL
        self.batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        self.concurrency_limit = model_semaphore
    
    async def check_model_availability(self) -> Dict[str, Any]:
        """Check if embedding model is available."""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.client.list),
                timeout=min(5, settings.OLLAMA_TIMEOUT),
            )
            models = []
            if hasattr(response, 'models'):
                models = [m.model if hasattr(m, 'model') else m.get('name', '') for m in response.models]
            elif isinstance(response, dict):
                models = [m.get('name', '') for m in response.get('models', [])]
            
            model_available = any(self.model in m for m in models)
            return {
                "model": self.model,
                "available": model_available,
                "error": None if model_available else f"Model {self.model} not found. Run: ollama pull {self.model}"
            }
        except Exception as e:
            return {
                "model": self.model,
                "available": False,
                "error": f"Cannot connect to Ollama: {type(e).__name__}. Ensure Ollama is running locally."
            }
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        async with self.concurrency_limit:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.embeddings,
                        model=self.model,
                        prompt=text,
                    ),
                    timeout=settings.OLLAMA_TIMEOUT,
                )
                if hasattr(response, 'embedding'):
                    return response.embedding
                elif isinstance(response, dict):
                    return response.get('embedding', [])
                else:
                    raise ValueError("Invalid embedding response")
            except Exception as e:
                logger.error(f"Embedding generation failed: {type(e).__name__}")
                raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for text in batch:
                emb = await self.embed_text(text)
                embeddings.append(emb)
        return embeddings


embedding_service = EmbeddingService()
