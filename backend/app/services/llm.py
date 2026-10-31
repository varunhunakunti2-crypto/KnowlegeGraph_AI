"""LLM service using Ollama (qwen3:4b). No paid APIs."""
from typing import Dict, Any, Optional
import asyncio

from app.core.config import settings
from app.core.logging import logger
from app.core.resources import model_semaphore
from app.core import memory
from app.services.ollama_client import get_client


class LLMService:
    """Service for LLM inference via local Ollama."""

    def __init__(self):
        self.model = settings.OLLAMA_MODEL
        self.max_retries = max(1, settings.OLLAMA_MAX_RETRIES)
        self.concurrency_limit = model_semaphore

    @property
    def client(self):
        # Shared process-wide client; no duplicate connection pools.
        return get_client()

    async def check_model_availability(self) -> Dict[str, Any]:
        """Check if required model is available."""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.client.list),
                timeout=min(5, settings.OLLAMA_TIMEOUT),
            )
            models = []
            # Handle different response formats
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
    
    async def generate(self, prompt: str, system_prompt: str = None,
                       temperature: float = None, max_tokens: int = None,
                       response_format: Optional[Dict[str, Any]] = None,
                       think: Optional[bool] = None) -> str:
        """Generate a response with bounded concurrency and timeouts.

        ``response_format`` is passed to Ollama's structured-output ``format``
        parameter. Setting ``think=False`` disables Qwen3 reasoning for
        structured tasks so the token budget is spent on the answer.

        The memory guard runs before the request (evicting another app model or
        unloading under pressure) and after it (releasing memory when the host
        is tight). Out-of-memory errors are converted into a clear, actionable
        exception instead of a generic 500.
        """
        if temperature is None:
            temperature = settings.OLLAMA_TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.OLLAMA_MAX_TOKENS

        async with self.concurrency_limit:
            await asyncio.to_thread(memory.before_model_call, self.model)
            try:
                for attempt in range(self.max_retries):
                    try:
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": prompt})

                        chat_kwargs: Dict[str, Any] = {
                            "model": self.model,
                            "messages": messages,
                            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens,
                                "num_ctx": settings.MODEL_CONTEXT_LENGTH,
                                # Smaller prompt batches reduce peak VRAM.
                                "num_batch": settings.OLLAMA_NUM_BATCH,
                            },
                        }
                        if response_format is not None:
                            chat_kwargs["format"] = response_format
                        if think is not None:
                            chat_kwargs["think"] = think

                        # The Ollama client is synchronous. Run it in a worker
                        # thread so a slow local model cannot block the FastAPI
                        # event loop, and bound each attempt with a timeout.
                        response = await asyncio.wait_for(
                            asyncio.to_thread(self.client.chat, **chat_kwargs),
                            timeout=settings.OLLAMA_TIMEOUT,
                        )

                        if hasattr(response, 'message'):
                            content = response.message.content
                        elif isinstance(response, dict):
                            content = response.get('message', {}).get('content', '')
                        else:
                            content = str(response)

                        if not content or not content.strip():
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(1)
                                continue
                            raise ValueError("Empty response from model")

                        return content
                    except memory.InsufficientMemoryError:
                        raise
                    except asyncio.TimeoutError:
                        # wait_for cannot stop the worker thread. Retrying would
                        # leave two requests racing for the same model/VRAM.
                        logger.error("LLM generation timed out; not retrying")
                        raise
                    except Exception as e:
                        if memory.is_out_of_memory_error(e):
                            # Retrying an allocation failure only wastes time.
                            logger.error("LLM generation ran out of memory")
                            raise memory.out_of_memory_error(e, "answer generation") from e
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        logger.error(f"LLM generation failed: {type(e).__name__}")
                        raise
            finally:
                await asyncio.to_thread(memory.after_model_call)


llm_service = LLMService()
