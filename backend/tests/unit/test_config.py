"""Tests for configuration."""
import pytest
from app.core.config import Settings


@pytest.mark.unit
def test_settings_defaults():
    """Test settings have correct defaults."""
    settings = Settings()
    assert settings.OLLAMA_MODEL == "qwen3:4b"
    assert settings.OLLAMA_EMBED_MODEL == "nomic-embed-text"
    assert settings.EMBEDDING_DIMENSION == 768
    assert settings.BACKEND_HOST == "127.0.0.1"


@pytest.mark.unit
def test_cors_origins_parsing():
    """Test CORS origins parsing."""
    settings = Settings(BACKEND_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000")
    origins = settings.get_cors_origins()
    assert len(origins) == 2
    assert "http://localhost:3000" in origins


@pytest.mark.unit
def test_low_memory_profile_applies_small_safe_defaults():
    settings = Settings(MEMORY_PROFILE="low-memory", _env_file=None)

    assert settings.MODEL_CONTEXT_LENGTH == 2048
    assert settings.OLLAMA_MAX_TOKENS == 384
    assert settings.MAX_CONCURRENT_REQUESTS == 1
    assert settings.OLLAMA_SINGLE_MODEL_RESIDENT is True
    assert settings.INDEX_BATCH_SIZE == 4
    assert settings.MAX_CONTEXT_TOKENS <= settings.MODEL_CONTEXT_LENGTH


@pytest.mark.unit
def test_explicit_values_override_low_memory_profile():
    settings = Settings(
        MEMORY_PROFILE="low-memory",
        MODEL_CONTEXT_LENGTH=8192,
        MAX_CONCURRENT_REQUESTS=2,
        _env_file=None,
    )

    assert settings.MODEL_CONTEXT_LENGTH == 8192
    assert settings.MAX_CONCURRENT_REQUESTS == 2


@pytest.mark.unit
def test_invalid_memory_profile_is_rejected():
    with pytest.raises(Exception):
        Settings(MEMORY_PROFILE="turbo", _env_file=None)
