"""Regression tests for schema-constrained graph extraction."""
from unittest.mock import Mock, patch

import pytest

from app.services.graph_extraction import GraphExtractionService
from app.services.llm import LLMService


@pytest.mark.unit
@patch("app.services.llm.Client")
async def test_llm_generate_forwards_structured_output_options(mock_client):
    """Ollama must receive the JSON schema and the no-thinking flag."""
    mock_instance = Mock()
    mock_response = Mock()
    mock_response.message = Mock()
    mock_response.message.content = "{}"
    mock_instance.chat.return_value = mock_response
    mock_client.return_value = mock_instance

    service = LLMService()
    await service.generate(
        "prompt",
        response_format={"type": "object"},
        think=False,
    )

    kwargs = mock_instance.chat.call_args.kwargs
    assert kwargs["format"] == {"type": "object"}
    assert kwargs["think"] is False


@pytest.mark.unit
async def test_graph_extraction_uses_constrained_json(monkeypatch):
    """Extraction must use the approved schema and disable reasoning."""
    service = GraphExtractionService()
    captured = {}

    async def fake_generate(prompt, **kwargs):
        captured.update(kwargs)
        return '{"entities": [], "relationships": []}'

    monkeypatch.setattr(service.llm, "generate", fake_generate)

    result = await service.extract_from_chunk(
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "page_number": 1,
            "chunk_text": "Ada Lovelace worked on the Analytical Engine.",
        }
    )

    assert result.entities == []
    assert captured["think"] is False

    schema = captured["response_format"]
    assert schema["required"] == ["entities", "relationships"]
    entity_schema = schema["properties"]["entities"]["items"]
    relationship_schema = schema["properties"]["relationships"]["items"]
    assert "Person" in entity_schema["properties"]["type"]["enum"]
    assert "CREATED" in relationship_schema["properties"]["predicate"]["enum"]
    assert entity_schema["required"] == [
        "name",
        "type",
        "confidence",
        "evidence",
    ]
