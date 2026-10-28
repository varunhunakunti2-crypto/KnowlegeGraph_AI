"""Tests for graph schema validation."""
import pytest
from app.schemas.graph import Entity, Relationship, GraphExtractionResult
from pydantic import ValidationError


@pytest.mark.unit
def test_valid_entity():
    """Test valid entity creation."""
    entity = Entity(
        name="John Doe",
        type="Person",
        aliases=["JD"],
        confidence=0.9,
        source_chunk_id="chunk-123",
        source_document_id="doc-456",
        page_number=1,
        evidence="John Doe is mentioned here."
    )
    assert entity.name == "John Doe"
    assert entity.type == "Person"


@pytest.mark.unit
def test_invalid_entity_type():
    """Test invalid entity type is rejected."""
    with pytest.raises(ValidationError):
        Entity(
            name="Test",
            type="InvalidType",
            aliases=[],
            confidence=0.9,
            source_chunk_id="chunk-123",
            source_document_id="doc-456",
            page_number=1,
            evidence="test"
        )


@pytest.mark.unit
def test_entity_name_normalization_logic():
    """Test entity name normalization."""
    from app.db.graph import GraphDB
    db = GraphDB()
    assert db._normalize_name("  John Doe  ") == "john doe"
    entity_id = db._create_entity_id("John Doe", "Person")
    assert len(entity_id) == 16


@pytest.mark.unit
def test_valid_relationship():
    """Test valid relationship creation."""
    rel = Relationship(
        subject="John",
        predicate="WORKS_FOR",
        object="Acme",
        confidence=0.8,
        source_chunk_id="chunk-123",
        source_document_id="doc-456",
        page_number=1,
        evidence="John works for Acme."
    )
    assert rel.subject == "John"
    assert rel.predicate == "WORKS_FOR"


@pytest.mark.unit
def test_graph_extraction_result():
    """Test graph extraction result validation."""
    result = GraphExtractionResult(entities=[], relationships=[])
    assert len(result.entities) == 0
    assert len(result.relationships) == 0
