"""Retrieval schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional


class Evidence(BaseModel):
    """Evidence object with provenance."""
    source_type: str = Field(..., pattern="^(vector|graph)$")
    chunk_id: Optional[str] = None
    document_id: str
    page_number: int
    text: str
    score: float
    entity_id: Optional[str] = None
    relationship_info: Optional[dict] = None


class RetrievalResult(BaseModel):
    """Retrieval result."""
    evidence: List[Evidence]
    has_evidence: bool
    context: str
    total_tokens_estimate: int
