"""Graph API routes."""
import asyncio

from fastapi import APIRouter, HTTPException, status, Query
from app.db.graph import graph_db
from app.db.graph_explore import graph_explorer
from app.db.chunks import ChunkDB
from app.db.documents import DocumentDB
from app.services.graph_extraction import graph_extraction_service
from app.schemas.graph_explore import GraphResponse, EntitySearchResult, EntityDetail
from app.core.config import settings
from app.core.logging import logger
from app.core.security import (
    SecurityValidationError,
    validate_identifier,
    validate_uuid,
)
from typing import Optional, List

router = APIRouter()


def _document_id_or_400(document_id: Optional[str]) -> Optional[str]:
    if document_id is None:
        return None
    try:
        return validate_uuid(document_id)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _entity_id_or_400(entity_id: str) -> str:
    try:
        return validate_identifier(entity_id, "entity ID")
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/", response_model=GraphResponse)
async def get_graph(document_id: Optional[str] = None, limit: int = Query(100, ge=1, le=200)):
    """Get graph nodes and edges for visualization."""
    document_id = _document_id_or_400(document_id)
    try:
        if document_id:
            graph = graph_explorer.get_document_graph(document_id, limit_nodes=limit)
        else:
            graph = graph_db.get_graph(document_id=document_id, limit=limit)
        return GraphResponse(
            nodes=graph["nodes"],
            edges=graph["edges"],
            total_nodes=len(graph["nodes"]),
            total_edges=len(graph["edges"])
        )
    except Exception as exc:
        logger.error(f"Graph retrieval failed: {type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Failed to retrieve graph")


@router.get("/entities/search", response_model=List[EntitySearchResult])
async def search_entities(q: str = Query(..., min_length=1, max_length=200),
                          node_type: Optional[str] = Query(None, max_length=50),
                          document_id: Optional[str] = None,
                          limit: int = Query(20, ge=1, le=100)):
    """Search for entities by name."""
    document_id = _document_id_or_400(document_id)
    try:
        results = graph_explorer.search_entities(q, node_type=node_type,
                                                document_id=document_id, limit=limit)
        return results
    except Exception as exc:
        logger.error(f"Entity search failed: {type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Search failed")


@router.get("/entities/{entity_id}", response_model=EntityDetail)
async def get_entity(entity_id: str, depth: int = Query(1, ge=1, le=3),
                     document_id: Optional[str] = None):
    """Get entity and its neighbors."""
    entity_id = _entity_id_or_400(entity_id)
    document_id = _document_id_or_400(document_id)
    try:
        neighbors = graph_explorer.get_entity_neighbors(
            entity_id, depth=depth, document_id=document_id
        )
        evidence = graph_explorer.get_entity_evidence(
            entity_id, document_id=document_id
        )

        # Get entity info
        nodes = neighbors.get("nodes", [])
        entity_node = next((n for n in nodes if n["id"] == entity_id), None)
        if not entity_node:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entity not found")

        return EntityDetail(
            id=entity_node["id"],
            label=entity_node["label"],
            type=entity_node["type"],
            aliases=entity_node.get("aliases", []),
            evidence=evidence,
            neighbors=[n for n in nodes if n["id"] != entity_id]
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Entity retrieval failed: {type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Failed to get entity")


@router.get("/entities/{entity_id}/neighbors")
async def get_neighbors(entity_id: str, depth: int = Query(1, ge=1, le=3),
                       node_type: Optional[str] = Query(None, max_length=50),
                       rel_type: Optional[str] = Query(None, max_length=50),
                       document_id: Optional[str] = None,
                       limit: int = Query(50, ge=1, le=100)):
    """Get entity neighbors with filters."""
    entity_id = _entity_id_or_400(entity_id)
    document_id = _document_id_or_400(document_id)
    try:
        result = graph_explorer.get_entity_neighbors(entity_id, depth=depth,
                                                    node_type=node_type,
                                                    rel_type=rel_type, limit=limit,
                                                    document_id=document_id)
        return result
    except Exception as exc:
        logger.error(f"Neighbor retrieval failed: {type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Failed to get neighbors")


@router.get("/edges/evidence")
async def get_edge_evidence(source_id: str, target_id: str,
                            document_id: Optional[str] = None,
                            limit: int = Query(10, ge=1, le=20)):
    """Get source evidence for an edge."""
    source_id = _entity_id_or_400(source_id)
    target_id = _entity_id_or_400(target_id)
    document_id = _document_id_or_400(document_id)
    try:
        evidence = graph_explorer.get_edge_evidence(
            source_id, target_id, limit=limit, document_id=document_id
        )
        return {"evidence": evidence}
    except Exception as exc:
        logger.error(f"Edge evidence retrieval failed: {type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Failed to get edge evidence")


@router.post("/extract/{document_id}")
async def extract_graph(document_id: str):
    """Extract entities and relationships from document chunks."""
    try:
        document_id = validate_uuid(document_id)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not DocumentDB.get_document(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        chunks = ChunkDB.get_chunks_by_document(document_id)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no chunks to extract",
            )

        truncated = len(chunks) > settings.MAX_GRAPH_EXTRACTION_CHUNKS
        chunks = chunks[:settings.MAX_GRAPH_EXTRACTION_CHUNKS]
        total_entities = 0
        total_rels = 0

        async def _extract_all():
            nonlocal total_entities, total_rels
            for chunk in chunks:
                extraction = await graph_extraction_service.extract_from_chunk(chunk)
                # Storing opens Neo4j sessions and performs many round trips;
                # keep it off the event loop.
                result = await asyncio.to_thread(
                    graph_db.store_extraction, document_id, extraction
                )
                total_entities += result["entities_stored"]
                total_rels += result["relationships_stored"]

        try:
            await asyncio.wait_for(
                _extract_all(),
                timeout=settings.GRAPH_EXTRACTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Graph extraction timed out after {settings.GRAPH_EXTRACTION_TIMEOUT_SECONDS}s"
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Graph extraction timed out",
            )

        return {
            "document_id": document_id,
            "entities_extracted": total_entities,
            "relationships_extracted": total_rels,
            "processed_chunks": len(chunks),
            "truncated": truncated,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Graph extraction failed: {type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Extraction failed")


@router.delete("/document/{document_id}")
async def delete_document_graph(document_id: str):
    """Delete graph evidence for a document."""
    try:
        document_id = validate_uuid(document_id)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        await asyncio.to_thread(graph_db.delete_document_graph, document_id)
        return {"message": "Graph data deleted successfully", "document_id": document_id}
    except Exception as exc:
        logger.error(f"Graph deletion failed: {type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Deletion failed")
