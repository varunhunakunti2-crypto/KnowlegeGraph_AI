"""LLM service using Ollama (qwen3:4b). No paid APIs."""
from ollama import Client
from app.core.config import settings
from app.core.logging import logger
from app.core.resources import model_semaphore
from typing import Dict, Any, Optional
import json
import asyncio
from pydantic import BaseModel, ValidationError


class LLMService:
    """Service for LLM inference via local Ollama."""
    
    def __init__(self):
        self.client = Client(host=settings.OLLAMA_HOST, timeout=settings.OLLAMA_TIMEOUT)
        self.model = settings.OLLAMA_MODEL
        self.max_retries = max(1, settings.OLLAMA_MAX_RETRIES)
        self.concurrency_limit = model_semaphore
    
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
                       temperature: float = None, max_tokens: int = None) -> str:
        """Generate response with bounded concurrency and timeouts."""
        if temperature is None:
            temperature = settings.OLLAMA_TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.OLLAMA_MAX_TOKENS
        
        async with self.concurrency_limit:
            for attempt in range(self.max_retries):
                try:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    # The Ollama client is synchronous. Run it in a worker
                    # thread so a slow local model cannot block the FastAPI
                    # event loop, and bound each attempt with a timeout.
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.chat,
                            model=self.model,
                            messages=messages,
                            options={
                                "temperature": temperature,
                                "num_predict": max_tokens,
                                "num_ctx": settings.MODEL_CONTEXT_LENGTH,
                            },
                        ),
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
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    logger.error(f"LLM generation failed: {type(e).__name__}")
                    raise


llm_service = LLMService()
