"""Graph extraction schemas."""
from pydantic import BaseModel, Field
from typing import List

# Approved vocabularies. Keeping these centralized means the LLM can never
# invent a label or relationship type that is later trusted by the graph layer.
APPROVED_ENTITY_TYPES = (
    "Person",
    "Organization",
    "Technology",
    "Concept",
    "Location",
    "Event",
)
APPROVED_PREDICATES = (
    "WORKS_FOR",
    "CREATED",
    "USES",
    "LOCATED_IN",
    "PART_OF",
    "RELATED_TO",
    "MENTIONS",
)

_ENTITY_TYPE_PATTERN = "^(?:" + "|".join(APPROVED_ENTITY_TYPES) + ")$"
_PREDICATE_PATTERN = "^(?:" + "|".join(APPROVED_PREDICATES) + ")$"


class Entity(BaseModel):
    """Extracted entity."""
    name: str = Field(..., min_length=1, max_length=200)
    type: str = Field(..., pattern=_ENTITY_TYPE_PATTERN)
    aliases: List[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk_id: str = Field(..., min_length=1, max_length=64)
    source_document_id: str = Field(..., min_length=1, max_length=64)
    page_number: int = Field(ge=0)
    evidence: str = Field(..., min_length=1, max_length=1000)


class Relationship(BaseModel):
    """Extracted relationship triple."""
    subject: str = Field(..., min_length=1, max_length=200)
    predicate: str = Field(..., pattern=_PREDICATE_PATTERN)
    object: str = Field(..., min_length=1, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk_id: str = Field(..., min_length=1, max_length=64)
    source_document_id: str = Field(..., min_length=1, max_length=64)
    page_number: int = Field(ge=0)
    evidence: str = Field(..., min_length=1, max_length=1000)


class GraphExtractionResult(BaseModel):
    """Result of graph extraction."""
    entities: List[Entity]
    relationships: List[Relationship]
