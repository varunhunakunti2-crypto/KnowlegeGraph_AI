"""Chat API routes with GraphRAG and answer generation."""
from fastapi import APIRouter, HTTPException, status
from app.retrieval.graphrag import graphrag_retriever
from app.generation.answer import answer_generator
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.db.documents import DocumentDB
from app.core.logging import logger
from app.core.memory import InsufficientMemoryError
from app.core.security import SecurityValidationError, validate_uuid

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with GraphRAG retrieval and answer generation."""
    document_id = request.document_id
    if document_id is not None:
        try:
            document_id = validate_uuid(document_id)
        except SecurityValidationError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        # A document-scoped chat must not proceed against a missing document;
        # this also avoids graph queries with an attacker-chosen identifier.
        try:
            if not DocumentDB.document_exists(document_id):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Document lookup failed: {type(exc).__name__}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to verify document",
            )

    try:
        # Retrieve evidence
        retrieval_result = await graphrag_retriever.retrieve(
            question=request.question,
            document_id=document_id
        )

        if not retrieval_result["has_evidence"]:
            return ChatResponse(
                question=request.question,
                answer="Insufficient evidence to answer this question. No relevant information was found in the indexed documents.",
                citations=[],
                evidence=[],
                has_evidence=False,
                retrieval_metadata={"method": "graphrag", "evidence_count": 0},
                message="Insufficient evidence"
            )

        # Generate answer with citations
        generation_result = await answer_generator.generate_answer(
            question=request.question,
            evidence=retrieval_result["evidence"]
        )

        # Get document names for citations
        citations = []
        for cit in generation_result["citations"]:
            try:
                # Only the filename is needed; do not load the pages snapshot.
                filename = DocumentDB.get_document_filename(cit["document_id"])
                citations.append(Citation(
                    chunk_id=cit["chunk_id"],
                    document_id=cit["document_id"],
                    document_name=filename,
                    page_number=cit["page_number"],
                    snippet=cit["snippet"]
                ))
            except Exception:
                citations.append(Citation(
                    chunk_id=cit["chunk_id"],
                    document_id=cit["document_id"],
                    document_name=None,
                    page_number=cit["page_number"],
                    snippet=cit["snippet"],
                ))

        return ChatResponse(
            question=request.question,
            answer=generation_result["answer"],
            citations=citations,
            evidence=generation_result["evidence_snippets"],
            has_evidence=True,
            retrieval_metadata={
                "method": "graphrag",
                "evidence_count": len(retrieval_result["evidence"]),
                "context_tokens": retrieval_result["total_tokens_estimate"],
                "vector_results": sum(1 for e in retrieval_result["evidence"] if e.get("source_type") == "vector"),
                "graph_results": sum(1 for e in retrieval_result["evidence"] if e.get("source_type") == "graph")
            },
            message=None
        )
    except HTTPException:
        raise
    except InsufficientMemoryError as exc:
        logger.error("Chat stopped: insufficient local memory")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Chat failed: {type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Chat failed")
