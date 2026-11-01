"""Runtime memory/GPU inspection and local model lifecycle management.

The target machine is a 6 GB RTX 4050 laptop with 16 GB of RAM. Ollama decides
where a model runs, so this module *measures* the result (``nvidia-smi`` and
``/api/ps``) instead of assuming CUDA is active. It also provides a small guard
that unloads the app's models when host memory is under pressure or when they
have been idle.

All functions are synchronous because they call subprocesses and the Ollama
HTTP API. Async callers should wrap them in ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional

import httpx

from app.core.config import settings
from app.core.logging import logger

try:  # psutil is a declared dependency, but keep imports defensive.
    import psutil
except Exception:  # pragma: no cover - only hit on a broken install
    psutil = None  # type: ignore[assignment]


class InsufficientMemoryError(RuntimeError):
    """Raised when the local model cannot be run safely on this machine."""


_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "outofmemoryerror",
    "not enough memory",
    "insufficient memory",
    "unable to allocate",
    "failed to allocate",
    "cuda error",
    "cuda malloc",
    "cudamalloc",
    "ggml_backend_cuda",
    "allocation failure",
)

_GPU_CACHE_SECONDS = 3.0
_gpu_cache: Dict[str, Any] = {"at": 0.0, "value": None}
_last_model_use = 0.0


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Return True when an Ollama/CUDA error looks like memory exhaustion."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


def _memory_hint() -> str:
    memory = system_memory()
    gpu = gpu_status()
    parts = []
    if memory.get("available_mb") is not None:
        parts.append(f"free system RAM {memory['available_mb']} MB")
    if gpu and gpu.get("free_mb") is not None:
        parts.append(f"free VRAM {gpu['free_mb']} MB")
    detail = f" ({', '.join(parts)})" if parts else ""
    return (
        f"Local model memory is insufficient{detail}. Close other GPU/RAM-heavy "
        "applications, set MEMORY_PROFILE=low-memory, lower MODEL_CONTEXT_LENGTH "
        "and OLLAMA_MAX_TOKENS, or use a smaller model."
    )


def out_of_memory_error(exc: BaseException, operation: str) -> InsufficientMemoryError:
    """Build a user-facing error for an Ollama allocation failure."""
    return InsufficientMemoryError(f"Out of memory during {operation}. {_memory_hint()}")


def system_memory() -> Dict[str, Any]:
    """Read host RAM. Values are in MB; percent is 0-100."""
    if psutil is None:
        return {"available": False, "reason": "psutil is not installed"}
    try:
        info = psutil.virtual_memory()
        return {
            "available": True,
            "total_mb": int(info.total / (1024 * 1024)),
            "available_mb": int(info.available / (1024 * 1024)),
            "used_mb": int((info.total - info.available) / (1024 * 1024)),
            "percent": round(float(info.percent), 1),
        }
    except Exception as exc:  # pragma: no cover - platform specific
        logger.debug(f"System memory probe failed: {type(exc).__name__}")
        return {"available": False, "reason": type(exc).__name__}


def _parse_gpu_line(line: str) -> Optional[Dict[str, Any]]:
    fields = [part.strip() for part in line.split(",")]
    if len(fields) < 7:
        return None

    def as_int(value: str) -> Optional[int]:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    total = as_int(fields[1])
    used = as_int(fields[2])
    free = as_int(fields[3])
    utilization = as_int(fields[4])
    temperature = as_int(fields[5])
    return {
        "available": True,
        "name": fields[0],
        "vram_total_mb": total,
        "vram_used_mb": used,
        "free_mb": free,
        "utilization_percent": utilization,
        "temperature_c": temperature,
        "driver_version": fields[6],
    }


def gpu_status(force: bool = False) -> Optional[Dict[str, Any]]:
    """Return a real GPU sample from nvidia-smi, or None when unavailable.

    The result is cached briefly so per-request checks do not spawn a process
    for every model call. ``force=True`` is used by the status endpoint and the
    benchmark.
    """
    now = time.monotonic()
    cached = _gpu_cache.get("value")
    if (
        not force
        and cached is not None
        and now - float(_gpu_cache.get("at", 0.0)) < _GPU_CACHE_SECONDS
    ):
        return cached

    executable = shutil.which("nvidia-smi")
    if not executable:
        _gpu_cache.update({"at": now, "value": None})
        return None

    command = [
        executable,
        "--query-gpu=name,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug(f"nvidia-smi probe failed: {type(exc).__name__}")
        _gpu_cache.update({"at": now, "value": None})
        return None

    if completed.returncode != 0 or not completed.stdout.strip():
        _gpu_cache.update({"at": now, "value": None})
        return None

    value = _parse_gpu_line(completed.stdout.strip().splitlines()[0])
    _gpu_cache.update({"at": now, "value": value})
    return value


def _same_model(left: str, right: str) -> bool:
    def normalize(name: str) -> str:
        name = str(name or "").strip().lower()
        return name[:-7] if name.endswith(":latest") else name

    return normalize(left) == normalize(right)


def app_model_names() -> List[str]:
    return [settings.OLLAMA_MODEL, settings.OLLAMA_EMBED_MODEL]


def loaded_models() -> Dict[str, Any]:
    """Return models currently resident in Ollama (from /api/ps)."""
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{settings.OLLAMA_HOST}/api/ps")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.debug(f"Ollama /api/ps unreachable: {type(exc).__name__}")
        return {"reachable": False, "models": []}

    models = []
    for item in payload.get("models", []) or []:
        size = item.get("size") or 0
        size_vram = item.get("size_vram") or 0
        models.append({
            "name": item.get("name") or item.get("model") or "",
            "size_mb": int(size / (1024 * 1024)) if size else None,
            "vram_mb": int(size_vram / (1024 * 1024)) if size_vram else None,
            "expires_at": item.get("expires_at"),
            "gpu_resident": bool(size_vram and size_vram > 0),
        })
    return {"reachable": True, "models": models}


def unload_model(model: str) -> bool:
    """Ask Ollama to unload one model immediately. Best effort."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.post(
                f"{settings.OLLAMA_HOST}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
            )
            response.raise_for_status()
        logger.info(f"Unloaded local model: {model}")
        return True
    except Exception as exc:
        logger.debug(f"Could not unload {model}: {type(exc).__name__}")
        return False


def unload_app_models(models: Optional[Iterable[str]] = None) -> List[str]:
    """Unload only this application's models, never unrelated Ollama models."""
    targets = list(models) if models is not None else app_model_names()
    unloaded = []
    for name in targets:
        if unload_model(name):
            unloaded.append(name)
    return unloaded


def _ensure_single_resident(model: str) -> None:
    """In low-memory mode, evict other app models before loading this one."""
    if not settings.OLLAMA_SINGLE_MODEL_RESIDENT:
        return
    state = loaded_models()
    if not state["reachable"]:
        return
    app_names = app_model_names()
    for item in state["models"]:
        name = item.get("name", "")
        if not name or _same_model(name, model):
            continue
        if any(_same_model(name, known) for known in app_names):
            unload_model(name)


def memory_pressure(ignore_gpu_when_loaded: bool = True) -> Dict[str, Any]:
    """Describe whether memory is tight right now.

    GPU free memory is only treated as pressure when no app model is resident:
    a loaded 4B model intentionally occupies most of a 6 GB card, and unloading
    it on every request would cause constant reloads.
    """
    memory = system_memory()
    gpu = gpu_status()

    ram_low = (
        memory.get("available_mb") is not None
        and memory["available_mb"] < settings.MIN_FREE_MEMORY_MB
    )
    gpu_low = (
        gpu is not None
        and gpu.get("free_mb") is not None
        and gpu["free_mb"] < settings.MIN_FREE_VRAM_MB
    )

    # Fast path: no need to ask Ollama which models are resident.
    if not ram_low and not gpu_low:
        return {
            "active": False,
            "reasons": [],
            "system_memory": memory,
            "gpu": gpu,
            "app_models_resident": False,
            "loaded_models": [],
        }

    state = loaded_models()
    resident = [m for m in state["models"] if m.get("name")]
    app_resident = any(
        _same_model(m["name"], known)
        for m in resident
        for known in app_model_names()
    )

    reasons = []
    if ram_low:
        reasons.append(
            f"system RAM available {memory['available_mb']} MB is below "
            f"MIN_FREE_MEMORY_MB={settings.MIN_FREE_MEMORY_MB}"
        )
    if gpu_low and not (ignore_gpu_when_loaded and app_resident):
        reasons.append(
            f"GPU free memory {gpu['free_mb']} MB is below "
            f"MIN_FREE_VRAM_MB={settings.MIN_FREE_VRAM_MB}"
        )

    return {
        "active": bool(reasons),
        "reasons": reasons,
        "system_memory": memory,
        "gpu": gpu,
        "app_models_resident": app_resident,
        "loaded_models": resident,
    }


def before_model_call(model: str) -> None:
    """Prepare the host before a generation or embedding request.

    - evicts the other app model in single-model mode,
    - unloads models under pressure,
    - fails fast (with actionable guidance) when memory is critically low.
    """
    global _last_model_use
    _last_model_use = time.monotonic()
    if not settings.MEMORY_GUARD_ENABLED:
        return

    _ensure_single_resident(model)

    pressure = memory_pressure()
    if pressure["active"]:
        # If the requested model is already resident, do not unload it just to
        # re-check free memory: the allocation is already paid and a reload
        # would be slower and can fail on an otherwise working request.
        state = loaded_models()
        resident = state["reachable"] and any(
            m.get("name") and _same_model(m["name"], model)
            for m in state["models"]
        )
        if resident:
            return
        if settings.OLLAMA_UNLOAD_ON_MEMORY_PRESSURE:
            unload_app_models()
            pressure = memory_pressure()

    available = pressure["system_memory"].get("available_mb")
    critical_floor = max(256, settings.MIN_FREE_MEMORY_MB // 2)
    if available is not None and available < critical_floor:
        raise InsufficientMemoryError(
            f"Not enough free system RAM to load {model}. {_memory_hint()}"
        )


def after_model_call() -> None:
    """Release models after a request when the host is under pressure."""
    global _last_model_use
    _last_model_use = time.monotonic()
    if not settings.MEMORY_GUARD_ENABLED:
        return
    if not settings.OLLAMA_UNLOAD_ON_MEMORY_PRESSURE:
        return
    try:
        if memory_pressure()["active"]:
            unload_app_models()
    except Exception as exc:  # pragma: no cover - guard must never break a request
        logger.debug(f"Post-request memory guard failed: {type(exc).__name__}")


async def idle_unload_monitor() -> None:
    """Background loop that unloads models after an idle period."""
    if not settings.MEMORY_GUARD_ENABLED or settings.OLLAMA_IDLE_UNLOAD_SECONDS <= 0:
        return

    interval = max(5, settings.MEMORY_CHECK_INTERVAL_SECONDS)
    while True:
        await asyncio.sleep(interval)
        idle_for = time.monotonic() - _last_model_use
        if idle_for < settings.OLLAMA_IDLE_UNLOAD_SECONDS:
            continue
        state = await asyncio.to_thread(loaded_models)
        if not state["reachable"] or not state["models"]:
            continue
        # Never unload models owned by another application, even after a long
        # idle period. Only names matching this app's configured models qualify.
        app_names = app_model_names()
        targets = [
            m["name"]
            for m in state["models"]
            if m.get("name")
            and any(_same_model(m["name"], known) for known in app_names)
        ]
        if not targets:
            continue
        logger.info(
            f"Unloading idle local models after {int(idle_for)}s "
            f"(OLLAMA_IDLE_UNLOAD_SECONDS={settings.OLLAMA_IDLE_UNLOAD_SECONDS})"
        )
        await asyncio.to_thread(unload_app_models, targets)


def runtime_status() -> Dict[str, Any]:
    """Full measured snapshot for /api/v1/system/status and the benchmark."""
    pressure = memory_pressure()
    loaded = loaded_models()
    # memory_pressure() uses a cheap fast path when nothing looks tight, so do
    # not trust its app_models_resident flag for the status endpoint. Derive it
    # from the /api/ps snapshot we already requested.
    resident = [m for m in loaded["models"] if m.get("name")]
    app_resident = any(
        _same_model(m["name"], known)
        for m in resident
        for known in app_model_names()
    )
    return {
        "profile": settings.MEMORY_PROFILE,
        "memory_guard_enabled": settings.MEMORY_GUARD_ENABLED,
        "system_memory": pressure["system_memory"],
        "gpu": gpu_status(force=True),
        "ollama": {
            "host": settings.OLLAMA_HOST,
            "reachable": loaded["reachable"],
            "loaded_models": loaded["models"],
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            "idle_unload_seconds": settings.OLLAMA_IDLE_UNLOAD_SECONDS,
            "single_model_resident": settings.OLLAMA_SINGLE_MODEL_RESIDENT,
        },
        "pressure": {
            "active": pressure["active"],
            "reasons": pressure["reasons"],
            "app_models_resident": app_resident,
        },
        "limits": {
            "model_context_length": settings.MODEL_CONTEXT_LENGTH,
            "max_output_tokens": settings.OLLAMA_MAX_TOKENS,
            "max_context_tokens": settings.MAX_CONTEXT_TOKENS,
            "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS,
            "max_concurrent_uploads": settings.MAX_CONCURRENT_UPLOADS,
            "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE,
            "index_batch_size": settings.INDEX_BATCH_SIZE,
            "batch_size": settings.BATCH_SIZE,
            "ollama_num_batch": settings.OLLAMA_NUM_BATCH,
            "graph_traversal_depth": settings.GRAPH_TRAVERSAL_DEPTH,
            "graph_traversal_depth_cap": settings.MAX_GRAPH_TRAVERSAL_DEPTH,
            "graph_limit": settings.MAX_GRAPH_LIMIT,
            "graph_extraction_chunks": settings.MAX_GRAPH_EXTRACTION_CHUNKS,
        },
    }
