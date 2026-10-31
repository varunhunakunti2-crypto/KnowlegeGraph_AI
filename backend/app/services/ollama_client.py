"""One shared Ollama client per backend process.

Creating a client per service is harmless for correctness, but a single client
keeps the connection pool and timeouts consistent. Ollama itself still loads one
model instance per model name; model residency is controlled by ``keep_alive``
and the memory guard in ``app.core.memory``.
"""
from __future__ import annotations

import threading

from ollama import Client

from app.core.config import settings

_client: Client | None = None
_lock = threading.Lock()


def get_client() -> Client:
    """Return the process-wide Ollama client."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = Client(
                    host=settings.OLLAMA_HOST,
                    timeout=settings.OLLAMA_TIMEOUT,
                )
    return _client


def reset_client() -> None:
    """Drop the cached client (used by tests and after config changes)."""
    global _client
    with _lock:
        _client = None
