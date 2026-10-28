"""Documents API routes."""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.documents import DocumentUploadResponse, DocumentMetadata, DocumentDetail
from app.ingestion.validator import PDFValidator, PDFValidationError
from app.ingestion.extractor import PDFExtractor
from app.db.documents import DocumentDB
from app.core.config import settings
import os
import uuid
import aiofiles
from pathlib import Path

router = APIRouter()


@router.on_event("startup")
async def init_db():
    try:
        DocumentDB.init_table()
    except Exception:
        pass


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    # Validate content type
    if file.content_type not in ["application/pdf", "application/x-pdf"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are supported."
        )
    
    # Check file extension
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension. Only .pdf files are supported."
        )
    
    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}.pdf"
    
    # Ensure upload directory exists
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / stored_filename
    
    # Prevent path traversal
    try:
        file_path.resolve().relative_to(upload_dir.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path"
        )
    
    try:
        # Save file
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file"
            )
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
        
        file_size = os.path.getsize(file_path)
        max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)"
            )
        
        # Validate PDF
        try:
            validation_result = PDFValidator.validate_upload(str(file_path))
            page_count = validation_result["page_count"]
        except PDFValidationError as e:
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Store in database
        try:
            DocumentDB.create_document(
                document_id=document_id,
                original_filename=file.filename,
                stored_filename=stored_filename,
                file_path=str(file_path),
                file_size=file_size,
                page_count=page_count,
                status="processing"
            )
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store document metadata"
            )
        
        # Extract text
        try:
            pages = PDFExtractor.extract_text(str(file_path))
            DocumentDB.update_document_status(document_id, "completed", pages=pages)
        except PDFValidationError as e:
            DocumentDB.update_document_status(document_id, "failed", error_message=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            DocumentDB.update_document_status(document_id, "failed", error_message=f"Extraction failed: {type(e).__name__}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract text"
            )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="completed",
            message="Document uploaded and processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed"
        )


@router.get("/", response_model=list[DocumentMetadata])
async def list_documents():
    """List all documents."""
    try:
        documents = DocumentDB.list_documents()
        return documents
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str):
    """Get document details with pages."""
    try:
        document = DocumentDB.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its files."""
    try:
        result = DocumentDB.delete_document(document_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        file_path = result["file_path"]
        # Safe cleanup - only delete if in upload dir
        try:
            upload_dir = Path(settings.UPLOAD_DIR).resolve()
            file_path_resolved = Path(file_path).resolve()
            file_path_resolved.relative_to(upload_dir)
            if file_path_resolved.exists():
                os.remove(file_path_resolved)
        except (ValueError, Exception):
            # Don't delete if outside upload dir or any error
            pass
        
        return {"message": "Document deleted successfully", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )
