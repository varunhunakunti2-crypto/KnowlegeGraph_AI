"""Memory guard, GPU parsing and actionable OOM errors."""
import pytest

from app.core import memory


@pytest.mark.unit
def test_parse_gpu_line_reads_real_nvidia_smi_fields():
    parsed = memory._parse_gpu_line(
        "NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 4000, 2141, 92, 67, 591.91"
    )
    assert parsed["name"].startswith("NVIDIA GeForce RTX 4050")
    assert parsed["vram_total_mb"] == 6141
    assert parsed["vram_used_mb"] == 4000
    assert parsed["free_mb"] == 2141
    assert parsed["utilization_percent"] == 92
    assert parsed["temperature_c"] == 67


@pytest.mark.unit
def test_parse_gpu_line_rejects_incomplete_output():
    assert memory._parse_gpu_line("NVIDIA GPU, 6141") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: out of memory",
        "failed to allocate 4 GiB",
        "ggml_backend_cuda: not enough memory",
        "OutOfMemoryError",
    ],
)
def test_out_of_memory_detection(message):
    assert memory.is_out_of_memory_error(RuntimeError(message))


@pytest.mark.unit
def test_non_memory_errors_are_not_misclassified():
    assert not memory.is_out_of_memory_error(RuntimeError("model not found"))
    assert not memory.is_out_of_memory_error(ValueError("invalid JSON"))


@pytest.mark.unit
def test_oom_error_message_is_actionable(monkeypatch):
    monkeypatch.setattr(
        memory, "system_memory", lambda: {"available": True, "available_mb": 300}
    )
    monkeypatch.setattr(
        memory,
        "gpu_status",
        lambda force=False: {"available": True, "free_mb": 200},
    )

    error = memory.out_of_memory_error(RuntimeError("CUDA out of memory"), "chat")
    text = str(error)

    assert "Out of memory during chat" in text
    assert "MEMORY_PROFILE=low-memory" in text
    assert "free system RAM 300 MB" in text
    assert "free VRAM 200 MB" in text


@pytest.mark.unit
def test_memory_pressure_reports_both_limits(monkeypatch):
    monkeypatch.setattr(
        memory, "system_memory", lambda: {"available": True, "available_mb": 200}
    )
    monkeypatch.setattr(
        memory,
        "gpu_status",
        lambda force=False: {"available": True, "free_mb": 100},
    )
    monkeypatch.setattr(memory, "loaded_models", lambda: {"reachable": True, "models": []})
    monkeypatch.setattr(memory.settings, "MIN_FREE_MEMORY_MB", 1024)
    monkeypatch.setattr(memory.settings, "MIN_FREE_VRAM_MB", 512)

    pressure = memory.memory_pressure()

    assert pressure["active"] is True
    assert len(pressure["reasons"]) == 2


@pytest.mark.unit
def test_gpu_pressure_is_ignored_while_an_app_model_is_resident(monkeypatch):
    monkeypatch.setattr(
        memory, "system_memory", lambda: {"available": True, "available_mb": 4096}
    )
    monkeypatch.setattr(
        memory,
        "gpu_status",
        lambda force=False: {"available": True, "free_mb": 100},
    )
    monkeypatch.setattr(
        memory,
        "loaded_models",
        lambda: {
            "reachable": True,
            "models": [{"name": memory.settings.OLLAMA_MODEL}],
        },
    )
    monkeypatch.setattr(memory.settings, "MIN_FREE_MEMORY_MB", 1024)
    monkeypatch.setattr(memory.settings, "MIN_FREE_VRAM_MB", 512)

    assert memory.memory_pressure()["active"] is False


@pytest.mark.unit
def test_before_model_call_unloads_then_fails_fast_when_ram_is_critical(monkeypatch):
    unloaded = []
    monkeypatch.setattr(memory, "_ensure_single_resident", lambda _model: None)
    monkeypatch.setattr(
        memory,
        "memory_pressure",
        lambda: {
            "active": True,
            "system_memory": {"available": True, "available_mb": 200},
            "gpu": None,
            "reasons": ["low ram"],
        },
    )
    monkeypatch.setattr(memory, "unload_app_models", lambda: unloaded.append(True))
    monkeypatch.setattr(
        memory, "loaded_models", lambda: {"reachable": True, "models": []}
    )
    monkeypatch.setattr(memory.settings, "MIN_FREE_MEMORY_MB", 1024)
    monkeypatch.setattr(memory.settings, "MEMORY_GUARD_ENABLED", True)
    monkeypatch.setattr(memory.settings, "OLLAMA_UNLOAD_ON_MEMORY_PRESSURE", True)

    with pytest.raises(memory.InsufficientMemoryError):
        memory.before_model_call("qwen3:4b")

    assert unloaded == [True]


@pytest.mark.unit
def test_before_model_call_keeps_a_resident_model_under_pressure(monkeypatch):
    """A model that is already loaded is usable even when free RAM looks low."""
    unloaded = []
    monkeypatch.setattr(memory, "_ensure_single_resident", lambda _model: None)
    monkeypatch.setattr(
        memory,
        "memory_pressure",
        lambda: {
            "active": True,
            "system_memory": {"available": True, "available_mb": 200},
            "gpu": None,
            "reasons": ["low ram"],
        },
    )
    monkeypatch.setattr(
        memory,
        "loaded_models",
        lambda: {
            "reachable": True,
            "models": [{"name": memory.settings.OLLAMA_EMBED_MODEL}],
        },
    )
    monkeypatch.setattr(memory, "unload_app_models", lambda: unloaded.append(True))
    monkeypatch.setattr(memory.settings, "MEMORY_GUARD_ENABLED", True)
    monkeypatch.setattr(memory.settings, "OLLAMA_UNLOAD_ON_MEMORY_PRESSURE", True)

    memory.before_model_call(memory.settings.OLLAMA_EMBED_MODEL)

    assert unloaded == []


@pytest.mark.unit
async def test_idle_unload_only_targets_app_models(monkeypatch):
    sleeps = {"count": 0}

    async def fake_sleep(_seconds):
        sleeps["count"] += 1
        if sleeps["count"] > 1:
            raise RuntimeError("stop test loop")

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    monkeypatch.setattr(memory.settings, "MEMORY_GUARD_ENABLED", True)
    monkeypatch.setattr(memory.settings, "OLLAMA_IDLE_UNLOAD_SECONDS", 120)
    monkeypatch.setattr(memory.settings, "MEMORY_CHECK_INTERVAL_SECONDS", 5)
    monkeypatch.setattr(memory, "_last_model_use", -100000.0)
    monkeypatch.setattr(
        memory,
        "loaded_models",
        lambda: {
            "reachable": True,
            "models": [{"name": "qwen3:4b"}, {"name": "unrelated-model:latest"}],
        },
    )
    unloaded = []
    monkeypatch.setattr(
        memory, "unload_app_models", lambda models: unloaded.extend(models)
    )

    with pytest.raises(RuntimeError, match="stop test loop"):
        await memory.idle_unload_monitor()

    assert unloaded == ["qwen3:4b"]


@pytest.mark.unit
def test_runtime_status_reports_resident_app_model(monkeypatch):
    monkeypatch.setattr(
        memory, "system_memory", lambda: {"available": True, "available_mb": 4096}
    )
    monkeypatch.setattr(memory, "gpu_status", lambda force=False: None)
    monkeypatch.setattr(
        memory,
        "loaded_models",
        lambda: {
            "reachable": True,
            "models": [{"name": "qwen3:4b", "vram_mb": 3000}],
        },
    )

    status = memory.runtime_status()

    assert status["pressure"]["app_models_resident"] is True
    assert status["ollama"]["loaded_models"][0]["name"] == "qwen3:4b"
