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
