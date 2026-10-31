"""Embedding service using Ollama (nomic-embed-text). No paid APIs."""
from typing import List, Dict, Any
import asyncio

from app.core.config import settings
from app.core.logging import logger
from app.core.resources import model_semaphore
from app.core import memory
from app.services.ollama_client import get_client


class EmbeddingService:
    """Service for embeddings via local Ollama."""

    def __init__(self):
        self.model = settings.OLLAMA_EMBED_MODEL
        # The indexing layer caps this by BATCH_SIZE and INDEX_BATCH_SIZE too.
        self.batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        self.concurrency_limit = model_semaphore

    @property
    def client(self):
        return get_client()

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

    def _extract_embedding(self, response: Any) -> List[float]:
        if hasattr(response, 'embedding'):
            return list(response.embedding)
        if isinstance(response, dict):
            return list(response.get('embedding', []))
        raise ValueError("Invalid embedding response")

    async def embed_text(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        async with self.concurrency_limit:
            await asyncio.to_thread(memory.before_model_call, self.model)
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.embeddings,
                        model=self.model,
                        prompt=text,
                        keep_alive=settings.OLLAMA_KEEP_ALIVE,
                    ),
                    timeout=settings.OLLAMA_TIMEOUT,
                )
                return self._extract_embedding(response)
            except memory.InsufficientMemoryError:
                raise
            except Exception as e:
                if memory.is_out_of_memory_error(e):
                    logger.error("Embedding generation ran out of memory")
                    raise memory.out_of_memory_error(e, "embedding") from e
                logger.error(f"Embedding generation failed: {type(e).__name__}")
                raise
            finally:
                await asyncio.to_thread(memory.after_model_call)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in small requests.

        The batch endpoint (``/api/embed``) embeds several texts at once, which
        is much faster than one HTTP round trip per chunk while keeping the
        per-request batch small enough for a 6 GB GPU. Falls back to the
        single-text endpoint if the batch response is unusable.
        """
        if not texts:
            return []

        request_batch = max(
            1,
            min(self.batch_size, settings.BATCH_SIZE, settings.INDEX_BATCH_SIZE),
        )
        embeddings: List[List[float]] = []

        async with self.concurrency_limit:
            await asyncio.to_thread(memory.before_model_call, self.model)
            try:
                for start in range(0, len(texts), request_batch):
                    batch = texts[start:start + request_batch]
                    try:
                        response = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.client.embed,
                                model=self.model,
                                input=batch,
                                keep_alive=settings.OLLAMA_KEEP_ALIVE,
                            ),
                            timeout=settings.OLLAMA_TIMEOUT,
                        )
                        if hasattr(response, 'embeddings'):
                            batch_vectors = list(response.embeddings)
                        elif isinstance(response, dict):
                            batch_vectors = list(response.get('embeddings', []))
                        else:
                            batch_vectors = []

                        if len(batch_vectors) != len(batch):
                            raise ValueError("Embedding batch size mismatch")
                        embeddings.extend([list(vector) for vector in batch_vectors])
                    except memory.InsufficientMemoryError:
                        raise
                    except Exception as e:
                        if memory.is_out_of_memory_error(e):
                            raise memory.out_of_memory_error(e, "embedding") from e
                        logger.error(f"Batch embedding failed: {type(e).__name__}")
                        raise
            finally:
                await asyncio.to_thread(memory.after_model_call)

        return embeddings


embedding_service = EmbeddingService()
