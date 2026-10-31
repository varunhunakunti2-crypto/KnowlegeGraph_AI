"""Documents API routes."""
import asyncio
import os
from pathlib import Path
from typing import Optional

import aiofiles
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status

from app.schemas.documents import DocumentUploadResponse, DocumentMetadata, DocumentDetail
from app.ingestion.validator import PDFValidator, PDFValidationError
from app.ingestion.extractor import PDFExtractor
from app.ingestion.chunker import TextChunker
from app.db.documents import DocumentDB
from app.db.chunks import ChunkDB
from app.db.vectors import VectorDB
from app.db.graph import graph_db
from app.core.config import settings
from app.core.logging import logger
from app.core.resources import upload_semaphore
from app.core.memory import InsufficientMemoryError
from app.core.security import (
    SecurityValidationError,
    redact_local_paths,
    remove_file_within,
    safe_join_upload_path,
    sanitize_filename,
    validate_uuid,
)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024


class UploadTooLarge(Exception):
    """Raised while streaming when an upload exceeds the configured limit."""


async def _stream_upload_to_file(file: UploadFile, destination: Path, max_bytes: int) -> int:
    """Write an upload to disk without ever holding the whole file in RAM."""
    total = 0
    async with aiofiles.open(destination, "wb") as handle:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLarge()
            await handle.write(chunk)
    return total


def _allowed_extensions() -> set[str]:
    return {
        extension.strip().lower()
        for extension in settings.ALLOWED_EXTENSIONS.split(",")
        if extension.strip()
    }


def _record_failure(document_id: str, file_path: Optional[Path], message: str) -> None:
    """Mark a failed ingestion, remove its partial data and stored file."""
    safe_message = redact_local_paths(message, max_length=300)
    try:
        DocumentDB.update_document_status(
            document_id, "failed", error_message=safe_message
        )
    except Exception as exc:
        logger.error(f"Failed to record document failure: {type(exc).__name__}")
    try:
        ChunkDB.delete_chunks_for_document(document_id)
    except Exception as exc:
        logger.warning(f"Failed to clean failed chunks: {type(exc).__name__}")
    if file_path is not None:
        remove_file_within(settings.UPLOAD_DIR, file_path)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Reject obviously oversized bodies before reading any of them. The
    # streaming write below is the authoritative check because Content-Length
    # can be omitted or lie.
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes + MULTIPART_OVERHEAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)",
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Content-Length header",
            )

    try:
        original_filename = sanitize_filename(file.filename)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type")
    if Path(original_filename).suffix.lower() not in _allowed_extensions():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file extension")

    async with upload_semaphore:
        document_id = str(uuid.uuid4())
        stored_filename = f"{document_id}.pdf"
        upload_dir = Path(settings.UPLOAD_DIR).resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_path = safe_join_upload_path(upload_dir, stored_filename)
            temp_path = safe_join_upload_path(upload_dir, f"{stored_filename}.part")
        except SecurityValidationError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file path")

        document_created = False
        try:
            try:
                file_size = await _stream_upload_to_file(file, temp_path, max_bytes)
            except UploadTooLarge:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)",
                )
            finally:
                await file.close()

            if file_size == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

            # Atomic rename avoids leaving a half-written PDF behind if the
            # process crashes mid-upload.
            os.replace(temp_path, file_path)

            try:
                validation_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        PDFValidator.validate_upload,
                        str(file_path),
                        file.content_type,
                        settings.MAX_UPLOAD_SIZE_MB,
                        settings.MAX_PDF_PAGES,
                        settings.ALLOW_PDF_ACTIVE_CONTENT,
                    ),
                    timeout=settings.UPLOAD_PROCESS_TIMEOUT_SECONDS,
                )
                page_count = validation_result["page_count"]
            except asyncio.TimeoutError:
                remove_file_within(upload_dir, file_path)
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="PDF validation timed out",
                )
            except PDFValidationError as exc:
                remove_file_within(upload_dir, file_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=redact_local_paths(str(exc)),
                )

            try:
                DocumentDB.create_document(
                    document_id,
                    original_filename,
                    stored_filename,
                    str(file_path),
                    file_size,
                    page_count,
                    "processing",
                )
                document_created = True
            except Exception as exc:
                logger.error(f"Failed to store document metadata: {type(exc).__name__}")
                remove_file_within(upload_dir, file_path)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to store metadata",
                )

            try:
                pages = await asyncio.wait_for(
                    asyncio.to_thread(
                        PDFExtractor.extract_text,
                        str(file_path),
                        settings.MAX_PDF_PAGES,
                        settings.MAX_EXTRACTED_TEXT_CHARS,
                    ),
                    timeout=settings.UPLOAD_PROCESS_TIMEOUT_SECONDS,
                )

                chunker = TextChunker(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                )
                chunks = chunker.chunk_pages(pages)
                if not chunks:
                    raise PDFValidationError("No extractable text found in PDF")
                if len(chunks) > settings.MAX_CHUNKS_PER_DOCUMENT:
                    raise PDFValidationError(
                        "Document produces too many chunks for this machine"
                    )

                DocumentDB.update_document_status(document_id, "completed", pages=pages)
                ChunkDB.create_chunks_idempotent(document_id, chunks, pages)
            except asyncio.TimeoutError:
                _record_failure(document_id, file_path, "Processing timed out")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Document processing timed out",
                )
            except PDFValidationError as exc:
                _record_failure(document_id, file_path, str(exc))
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=redact_local_paths(str(exc)),
                )
            except Exception as exc:
                logger.error(f"Document processing failed: {type(exc).__name__}")
                _record_failure(
                    document_id,
                    file_path,
                    f"Processing failed: {type(exc).__name__}",
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Processing failed",
                )

            return DocumentUploadResponse(
                document_id=document_id,
                filename=original_filename,
                status="completed",
                message="Document uploaded and processed",
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Upload failed: {type(exc).__name__}")
            if document_created:
                _record_failure(document_id, file_path, "Upload failed")
            else:
                remove_file_within(upload_dir, temp_path)
                remove_file_within(upload_dir, file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Upload failed",
            )
        finally:
            # The .part file should only exist before the atomic rename.
            if temp_path.exists():
                remove_file_within(upload_dir, temp_path)


@router.post("/{document_id}/index")
async def index_document(document_id: str):
    """Index document chunks for vector search."""
    try:
        document_id = validate_uuid(document_id)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not DocumentDB.document_exists(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        result = await VectorDB.index_document_chunks(document_id)
        status_info = VectorDB.get_indexing_status(document_id)
        return {"document_id": document_id, "indexing_result": result, "status": status_info}
    except InsufficientMemoryError as exc:
        logger.error("Indexing stopped: insufficient local memory")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Indexing failed: {type(exc).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Indexing failed",
        )


@router.get("/", response_model=list[DocumentMetadata])
async def list_documents():
    try:
        return DocumentDB.list_documents()
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve documents")


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str):
    try:
        document_id = validate_uuid(document_id)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        document = DocumentDB.get_document(document_id)
        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve document")


@router.get("/{document_id}/index/status")
async def get_index_status(document_id: str):
    try:
        document_id = validate_uuid(document_id)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not DocumentDB.document_exists(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        return VectorDB.get_indexing_status(document_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get status")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document consistently across files, chunks, embeddings and graph."""
    try:
        document_id = validate_uuid(document_id)
    except SecurityValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not DocumentDB.document_exists(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    warnings = []

    # Graph deletion is best-effort: a local dev machine may not have Neo4j
    # running, and SQL/file cleanup should still succeed. The warning is
    # explicit so callers know graph evidence may need a retry.
    graph_deleted = True
    try:
        await asyncio.to_thread(graph_db.delete_document_graph, document_id)
    except Exception as exc:
        graph_deleted = False
        warnings.append("Graph evidence cleanup incomplete")
        logger.warning(f"Graph cleanup failed: {type(exc).__name__}")

    try:
        result = DocumentDB.delete_document(document_id)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Document deletion failed: {type(exc).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )

    file_removed = remove_file_within(settings.UPLOAD_DIR, result["file_path"])
    if not file_removed:
        warnings.append("Stored file was already missing or outside the upload directory")

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "graph_deleted": graph_deleted,
        "file_removed": file_removed,
        "warnings": warnings,
    }
