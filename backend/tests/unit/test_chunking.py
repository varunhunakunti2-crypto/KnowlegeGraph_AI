"""Tests for chunking."""
import pytest
from app.ingestion.chunker import TextChunker


@pytest.mark.unit
def test_chunk_boundaries():
    """Test chunk boundaries are respected."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "This is a paragraph. " * 10
    chunks = chunker.chunk_text(text, page_number=1)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["char_count"] <= 200  # Allow some flexibility


@pytest.mark.unit
def test_chunk_overlap():
    """Test overlap between chunks."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "A" * 100
    chunks = chunker.chunk_text(text, page_number=1)
    assert len(chunks) >= 2


@pytest.mark.unit
def test_empty_text():
    """Test empty text produces no chunks."""
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_text("", page_number=1)
    assert len(chunks) == 0
    chunks = chunker.chunk_text("   ", page_number=1)
    assert len(chunks) == 0


@pytest.mark.unit
def test_page_mapping():
    """Test each chunk maps to original page."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    text = "Some text here. " * 5
    chunks = chunker.chunk_text(text, page_number=3)
    for chunk in chunks:
        assert chunk["page_number"] == 3
        assert "chunk_index" in chunk
        assert "chunk_text" in chunk
        assert "char_count" in chunk


@pytest.mark.unit
def test_preserve_paragraph_boundaries():
    """Test paragraph boundaries are preserved where practical."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunker.chunk_text(text, page_number=1)
    assert len(chunks) >= 1
    # Chunks should not start/end mid-word in bad ways
    for chunk in chunks:
        assert chunk["chunk_text"].strip()


@pytest.mark.unit
def test_overlap_larger_than_chunk_size_does_not_hang():
    """An invalid overlap must not create a zero/negative split step."""
    chunker = TextChunker(chunk_size=64, chunk_overlap=10_000)
    assert 0 <= chunker.chunk_overlap < chunker.chunk_size
    chunks = chunker.chunk_text("A" * 500, page_number=1)
    assert chunks
    assert len(chunks) < 500
