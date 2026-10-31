"""Tests for LLM service with mocked responses."""
import asyncio
import time

import pytest
from unittest.mock import Mock, patch

from app.services.llm import LLMService
from app.services.embeddings import EmbeddingService


@pytest.fixture(autouse=True)
def _disable_memory_guard(monkeypatch):
    """Unit tests must not probe nvidia-smi or the Ollama /api/ps endpoint."""
    from app.services import llm, embeddings

    for module in (llm, embeddings):
        monkeypatch.setattr(module.memory, "before_model_call", lambda *_: None)
        monkeypatch.setattr(module.memory, "after_model_call", lambda *_: None)


@pytest.mark.unit
@patch('app.services.llm.get_client')
async def test_llm_check_model_availability_missing(mock_get_client):
    """Test model availability check when model missing."""
    mock_instance = Mock()
    mock_instance.list.return_value = {'models': [{'name': 'other-model'}]}
    mock_get_client.return_value = mock_instance

    service = LLMService()
    result = await service.check_model_availability()
    assert result['available'] is False
    assert 'not found' in result['error'].lower()


@pytest.mark.unit
@patch('app.services.embeddings.get_client')
async def test_embedding_check_model_availability_missing(mock_get_client):
    """Test embedding model availability check when model missing."""
    mock_instance = Mock()
    mock_instance.list.return_value = {'models': [{'name': 'other-model'}]}
    mock_get_client.return_value = mock_instance

    service = EmbeddingService()
    result = await service.check_model_availability()
    assert result['available'] is False
    assert 'not found' in result['error'].lower()


@pytest.mark.unit
@patch('app.services.llm.get_client')
async def test_model_availability_check_is_time_bounded(mock_get_client, monkeypatch):
    """A hung Ollama process must not block the event loop indefinitely."""
    from app.services import llm

    monkeypatch.setattr(llm.settings, "OLLAMA_TIMEOUT", 0.01)
    mock_instance = Mock()
    mock_instance.list.side_effect = lambda: time.sleep(0.5)
    mock_get_client.return_value = mock_instance

    result = await LLMService().check_model_availability()

    assert result["available"] is False
    assert "Cannot connect" in result["error"]


@pytest.mark.unit
@patch('app.services.llm.get_client')
async def test_llm_generate_with_mock(mock_get_client):
    """Test LLM generation with mocked response."""
    mock_instance = Mock()
    mock_response = Mock()
    mock_response.message = Mock()
    mock_response.message.content = "Test response"
    mock_instance.chat.return_value = mock_response
    mock_get_client.return_value = mock_instance

    service = LLMService()
    result = await service.generate("test prompt")
    assert result == "Test response"


@pytest.mark.unit
@patch('app.services.llm.get_client')
async def test_llm_generate_passes_memory_limits(mock_get_client):
    """Context, output tokens, batch size and keep_alive must reach Ollama."""
    from app.core.config import settings

    mock_instance = Mock()
    mock_response = Mock()
    mock_response.message.content = "ok"
    mock_instance.chat.return_value = mock_response
    mock_get_client.return_value = mock_instance

    await LLMService().generate("prompt", max_tokens=123)

    kwargs = mock_instance.chat.call_args.kwargs
    assert kwargs["keep_alive"] == settings.OLLAMA_KEEP_ALIVE
    assert kwargs["options"]["num_predict"] == 123
    assert kwargs["options"]["num_ctx"] == settings.MODEL_CONTEXT_LENGTH
    assert kwargs["options"]["num_batch"] == settings.OLLAMA_NUM_BATCH


@pytest.mark.unit
@patch('app.services.llm.get_client')
async def test_llm_generate_maps_out_of_memory_error(mock_get_client):
    """CUDA allocation failures become a clear, actionable exception."""
    from app.core.memory import InsufficientMemoryError

    mock_instance = Mock()
    mock_instance.chat.side_effect = RuntimeError("CUDA error: out of memory")
    mock_get_client.return_value = mock_instance

    with pytest.raises(InsufficientMemoryError):
        await LLMService().generate("prompt")


@pytest.mark.unit
@patch('app.services.llm.get_client')
async def test_llm_timeout_does_not_start_a_second_request(
    mock_get_client, monkeypatch
):
    """A timed-out worker may still be running; do not race another request."""
    from app.services import llm

    monkeypatch.setattr(llm.settings, "OLLAMA_TIMEOUT", 0.01)
    mock_instance = Mock()
    mock_instance.chat.side_effect = lambda **_: time.sleep(0.2)
    mock_get_client.return_value = mock_instance

    with pytest.raises(asyncio.TimeoutError):
        await LLMService().generate("prompt")

    assert mock_instance.chat.call_count == 1


@pytest.mark.unit
@patch('app.services.embeddings.get_client')
async def test_embedding_with_mock(mock_get_client):
    """Test embedding generation with mocked response."""
    mock_instance = Mock()
    mock_response = Mock()
    mock_response.embedding = [0.1] * 768
    mock_instance.embeddings.return_value = mock_response
    mock_get_client.return_value = mock_instance

    service = EmbeddingService()
    result = await service.embed_text("test text")
    assert len(result) == 768
    assert isinstance(result, list)


@pytest.mark.unit
@patch('app.services.embeddings.get_client')
async def test_embed_batch_uses_small_requests(mock_get_client):
    """Batch embeddings are split into small /api/embed requests."""
    mock_instance = Mock()

    def fake_embed(model, input, keep_alive):
        response = Mock()
        response.embeddings = [[0.1] * 768 for _ in input]
        return response

    mock_instance.embed.side_effect = fake_embed
    mock_get_client.return_value = mock_instance

    service = EmbeddingService()
    service.batch_size = 2
    vectors = await service.embed_batch(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert mock_instance.embed.call_count == 3
    for call in mock_instance.embed.call_args_list:
        assert len(call.kwargs["input"]) <= 2
