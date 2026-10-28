"""Document schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str
    filename: str
    status: str
    message: str


class PageContent(BaseModel):
    """Page content with metadata."""
    page_number: int
    text: str
    char_count: int


class DocumentMetadata(BaseModel):
    """Document metadata."""
    document_id: str
    original_filename: str
    file_size: int
    page_count: int
    status: str
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class DocumentDetail(BaseModel):
    """Document detail with pages."""
    document_id: str
    original_filename: str
    file_size: int
    page_count: int
    status: str
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    pages: List[PageContent] = []
