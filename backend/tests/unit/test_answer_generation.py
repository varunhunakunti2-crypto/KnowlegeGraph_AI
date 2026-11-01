"""Tests for answer generation."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.generation.answer import AnswerGenerator


@pytest.mark.unit
@patch('app.generation.answer.llm_service')
async def test_answer_with_insufficient_evidence(mock_llm):
    """Test answer when no evidence provided."""
    generator = AnswerGenerator()
    result = await generator.generate_answer("What is X?", [])
    assert "insufficient" in result["answer"].lower()
    assert result["citations"] == []


@pytest.mark.unit
@patch('app.generation.answer.llm_service')
async def test_answer_with_citations(mock_llm):
    """Test answer with valid citations."""
    mock_llm.generate = AsyncMock(return_value="Based on [chunk:abc123], the answer is yes.")
    
    generator = AnswerGenerator()
    evidence = [{
        "chunk_id": "abc123",
        "document_id": "doc1",
        "page_number": 1,
        "text": "The answer is yes."
    }]
    result = await generator.generate_answer("Question?", evidence)
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "abc123"
    # Qwen3 must not spend the output budget on hidden reasoning; otherwise
    # citation-heavy questions can return an empty answer.
    assert mock_llm.generate.await_args.kwargs["think"] is False


@pytest.mark.unit
def test_citation_validation_rejects_invalid():
    """Test that invalid citations are not included."""
    generator = AnswerGenerator()
    answer = "See [chunk:invalid123] for details."
    evidence = [{
        "chunk_id": "abc123",
        "document_id": "doc1",
        "page_number": 1,
        "text": "test"
    }]
    citations = generator._extract_citations(answer, evidence)
    assert len(citations) == 0


@pytest.mark.unit
def test_system_prompt_separate():
    """Test system prompt is separate from user content."""
    generator = AnswerGenerator()
    system = generator._build_system_prompt()
    user = generator._build_user_prompt("question", [])
    assert "ONLY using the provided evidence" in system
    assert "question" not in system
    assert "Evidence:" in user
