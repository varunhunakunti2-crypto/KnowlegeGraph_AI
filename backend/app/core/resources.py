"""Bounded concurrency primitives for a 16 GB development machine.

All model and upload work goes through a shared limiter per process. Semaphores
are created lazily so importing this module in tests (where several event loops
may exist) does not bind to a stale loop.
"""
from __future__ import annotations

import asyncio

from app.core.config import settings


class LazySemaphore:
    """A semaphore that is (re)created for the running event loop."""

    def __init__(self, limit: int):
        self._limit = max(1, int(limit))
        self._semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self._limit)
            self._loop = loop
        return self._semaphore

    async def __aenter__(self):
        await self._get().acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._get().release()
        return False


# Shared GPU/RAM budget: generation and embeddings compete for the same model
# host, so one limit protects both. Uploads get their own limit because PDF
# parsing is CPU/RAM bound and independent of Ollama.
model_semaphore = LazySemaphore(settings.MAX_CONCURRENT_REQUESTS)
upload_semaphore = LazySemaphore(settings.MAX_CONCURRENT_UPLOADS)
