"""Graph exploration schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class NodeResponse(BaseModel):
    """Graph node."""
    id: str
    label: str
    type: str
    aliases: List[str] = []
    evidence_count: Optional[int] = 0


class EdgeResponse(BaseModel):
    """Graph edge."""
    id: str
    source: str
    target: str
    label: str
    confidence: Optional[float] = None
    evidence: Optional[List[Dict[str, Any]]] = []


class GraphResponse(BaseModel):
    """Graph response."""
    nodes: List[NodeResponse]
    edges: List[EdgeResponse]
    total_nodes: int
    total_edges: int


class EntitySearchResult(BaseModel):
    """Entity search result."""
    id: str
    label: str
    type: str
    document_count: int = 0


class EntityDetail(BaseModel):
    """Entity detail with evidence."""
    id: str
    label: str
    type: str
    aliases: List[str] = []
    evidence: List[Dict[str, Any]] = []
    neighbors: List[NodeResponse] = []
