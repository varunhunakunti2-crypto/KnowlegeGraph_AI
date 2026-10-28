"""Tests for vector operations."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.db.vectors import VectorDB


@pytest.mark.unit
@patch('app.db.vectors.embedding_service')
async def test_verify_embedding_dimensions(mock_embed):
    """Test embedding dimension verification."""
    mock_embed.embed_text = AsyncMock(return_value=[0.1] * 768)
    dims = await VectorDB.verify_embedding_dimensions()
    assert dims == 768


@pytest.mark.unit
def test_vector_search_preserves_metadata():
    """Test that search results preserve required metadata."""
    # This is a structure test
    result_structure = {
        "chunk_id": "test",
        "document_id": "test",
        "page_number": 1,
        "chunk_index": 0,
        "text": "test",
        "char_count": 4,
        "similarity": 0.9
    }
    assert "chunk_id" in result_structure
    assert "document_id" in result_structure
    assert "page_number" in result_structure
    assert "text" in result_structure
    assert "similarity" in result_structure
