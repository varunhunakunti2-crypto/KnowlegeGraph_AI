"""Security-focused configuration tests."""
import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.mark.unit
def test_wildcard_cors_is_rejected():
    with pytest.raises(ValidationError):
        Settings(BACKEND_CORS_ORIGINS="*")


@pytest.mark.unit
def test_cors_origins_are_explicit_and_normalized():
    settings = Settings(
        BACKEND_CORS_ORIGINS="http://localhost:3000/, http://127.0.0.1:3000"
    )
    assert settings.get_cors_origins() == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


@pytest.mark.unit
def test_non_http_cors_origin_is_rejected():
    with pytest.raises(ValidationError):
        Settings(BACKEND_CORS_ORIGINS="file:///etc/passwd")


@pytest.mark.unit
def test_default_database_hosts_are_local():
    # Test the code defaults, not the developer's .env overrides.
    settings = Settings(_env_file=None)
    assert settings.NEO4J_URI.startswith("bolt://localhost")
    assert settings.POSTGRES_HOST == "localhost"
    assert settings.BACKEND_HOST == "127.0.0.1"


@pytest.mark.unit
def test_context_budget_cannot_exceed_model_window():
    settings = Settings(MODEL_CONTEXT_LENGTH=1024, MAX_CONTEXT_TOKENS=4096)
    assert settings.MAX_CONTEXT_TOKENS == 1024


@pytest.mark.unit
def test_graph_depth_is_clamped():
    settings = Settings(GRAPH_TRAVERSAL_DEPTH=99)
    assert settings.GRAPH_TRAVERSAL_DEPTH == settings.MAX_GRAPH_TRAVERSAL_DEPTH


@pytest.mark.unit
def test_chunk_overlap_cannot_swallow_chunk_size():
    settings = Settings(CHUNK_SIZE=512, CHUNK_OVERLAP=4000)
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE


@pytest.mark.unit
def test_database_timeouts_are_bounded():
    with pytest.raises(ValidationError):
        Settings(DB_CONNECT_TIMEOUT_SECONDS=0)
    with pytest.raises(ValidationError):
        Settings(GRAPH_EXTRACTION_TIMEOUT_SECONDS=0)
