"""Chat schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional

from app.core.config import settings


class ChatRequest(BaseModel):
    # A bounded prompt prevents accidental resource exhaustion and keeps the
    # question inside the model context budget.
    question: str = Field(..., min_length=1, max_length=settings.MAX_QUESTION_CHARS)
    document_id: Optional[str] = Field(default=None, max_length=64)


class Citation(BaseModel):
    """Citation to source."""
    chunk_id: str
    document_id: str
    document_name: Optional[str] = None
    page_number: int
    snippet: str


class ChatResponse(BaseModel):
    """Chat response with citations."""
    question: str
    answer: str
    citations: List[Citation] = []
    evidence: List[dict] = []
    has_evidence: bool
    retrieval_metadata: dict = {}
    message: Optional[str] = None
