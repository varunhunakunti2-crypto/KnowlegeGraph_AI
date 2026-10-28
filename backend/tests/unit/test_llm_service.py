"""Tests for LLM service with mocked responses."""
import time

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.llm import LLMService
from app.services.embeddings import EmbeddingService


@pytest.mark.unit
@patch('app.services.llm.Client')
async def test_llm_check_model_availability_missing(mock_client):
    """Test model availability check when model missing."""
    mock_instance = Mock()
    mock_instance.list.return_value = {'models': [{'name': 'other-model'}]}
    mock_client.return_value = mock_instance
    
    service = LLMService()
    result = await service.check_model_availability()
    assert result['available'] is False
    assert 'not found' in result['error'].lower()


@pytest.mark.unit
@patch('app.services.embeddings.Client')
async def test_embedding_check_model_availability_missing(mock_client):
    """Test embedding model availability check when missing."""
    mock_instance = Mock()
    mock_instance.list.return_value = {'models': [{'name': 'other-model'}]}
    mock_client.return_value = mock_instance
    
    service = EmbeddingService()
    result = await service.check_model_availability()
    assert result['available'] is False
    assert 'not found' in result['error'].lower()


@pytest.mark.unit
@patch('app.services.llm.Client')
async def test_model_availability_check_is_time_bounded(mock_client, monkeypatch):
    """A hung Ollama process must not block the event loop indefinitely."""
    from app.services import llm

    monkeypatch.setattr(llm.settings, "OLLAMA_TIMEOUT", 0.01)
    mock_instance = Mock()
    mock_instance.list.side_effect = lambda: time.sleep(0.5)
    mock_client.return_value = mock_instance

    result = await LLMService().check_model_availability()

    assert result["available"] is False
    assert "Cannot connect" in result["error"]


@pytest.mark.unit
@patch('app.services.llm.Client')
async def test_llm_generate_with_mock(mock_client):
    """Test LLM generation with mocked response."""
    mock_instance = Mock()
    mock_response = Mock()
    mock_response.message = Mock()
    mock_response.message.content = "Test response"
    mock_instance.chat.return_value = mock_response
    mock_client.return_value = mock_instance
    
    service = LLMService()
    result = await service.generate("test prompt")
    assert result == "Test response"


@pytest.mark.unit
@patch('app.services.embeddings.Client')
async def test_embedding_with_mock(mock_client):
    """Test embedding generation with mocked response."""
    mock_instance = Mock()
    mock_response = Mock()
    mock_response.embedding = [0.1] * 768
    mock_instance.embeddings.return_value = mock_response
    mock_client.return_value = mock_instance
    
    service = EmbeddingService()
    result = await service.embed_text("test text")
    assert len(result) == 768
    assert isinstance(result, list)
