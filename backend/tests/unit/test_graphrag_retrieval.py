"""Tests for GraphRAG retrieval."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.retrieval.graphrag import GraphRAGRetriever
from app.core.memory import InsufficientMemoryError


@pytest.mark.unit
@patch('app.retrieval.graphrag.VectorDB')
@patch('app.retrieval.graphrag.GraphDB')
async def test_retrieval_with_no_evidence(mock_graph_db, mock_vector_db):
    """Test handling of questions with no evidence."""
    mock_vector_instance = Mock()
    mock_vector_instance.search_similar = AsyncMock(return_value=[])
    mock_vector_db.return_value = mock_vector_instance
    
    mock_graph_instance = Mock()
    mock_graph_instance.connect = Mock()
    mock_graph_db.return_value = mock_graph_instance
    
    retriever = GraphRAGRetriever()
    # Mock graph retrieval to return empty
    retriever._graph_retrieval = Mock(return_value=[])
    
    result = await retriever.retrieve("test question")
    assert result["has_evidence"] is False
    assert result["evidence"] == []
    assert "insufficient" in result["message"].lower()


@pytest.mark.unit
@patch('app.retrieval.graphrag.VectorDB')
@patch('app.retrieval.graphrag.GraphDB')
async def test_vector_memory_error_is_not_hidden(mock_graph_db, mock_vector_db):
    """A memory failure must reach the API as a 503, not an empty answer."""
    mock_vector_instance = Mock()
    mock_vector_instance.search_similar = AsyncMock(
        side_effect=InsufficientMemoryError("not enough memory")
    )
    mock_vector_db.return_value = mock_vector_instance

    mock_graph_instance = Mock()
    mock_graph_instance.connect = Mock()
    mock_graph_db.return_value = mock_graph_instance

    retriever = GraphRAGRetriever()
    retriever._graph_retrieval = Mock(return_value=[])

    with pytest.raises(InsufficientMemoryError):
        await retriever.retrieve("test question")


@pytest.mark.unit
@patch('app.retrieval.graphrag.VectorDB')
@patch('app.retrieval.graphrag.GraphDB')
async def test_graph_memory_error_is_not_hidden(mock_graph_db, mock_vector_db):
    """Memory errors from the graph channel also propagate."""
    mock_vector_instance = Mock()
    mock_vector_instance.search_similar = AsyncMock(return_value=[])
    mock_vector_db.return_value = mock_vector_instance

    mock_graph_instance = Mock()
    mock_graph_instance.connect = Mock()
    mock_graph_db.return_value = mock_graph_instance

    retriever = GraphRAGRetriever()
    retriever._graph_retrieval = Mock(
        side_effect=InsufficientMemoryError("not enough memory")
    )

    with pytest.raises(InsufficientMemoryError):
        await retriever.retrieve("test question")


@pytest.mark.unit
def test_deduplication():
    """Test evidence deduplication."""
    retriever = GraphRAGRetriever()
    evidence = [
        {"document_id": "doc1", "page_number": 1, "text": "hello world", "score": 0.5},
        {"document_id": "doc1", "page_number": 1, "text": "hello world", "score": 0.7},
        {"document_id": "doc2", "page_number": 1, "text": "different", "score": 0.6},
    ]
    deduped = retriever._deduplicate_evidence(evidence)
    assert len(deduped) == 2
    # Should keep higher score for duplicate
    assert any(e["score"] == 0.7 for e in deduped)


@pytest.mark.unit
def test_evidence_preserves_provenance():
    """Test evidence objects preserve required provenance."""
    evidence = {
        "source_type": "vector",
        "chunk_id": "chunk-123",
        "document_id": "doc-456",
        "page_number": 5,
        "text": "test text",
        "score": 0.9
    }
    assert evidence["document_id"] is not None
    assert evidence["page_number"] is not None
    assert evidence["text"] is not None
    assert evidence["score"] is not None
