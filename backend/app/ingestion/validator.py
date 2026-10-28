"""PDF validation."""
import os
from typing import Optional

import fitz  # PyMuPDF

from app.core.config import settings


class PDFValidationError(Exception):
    """PDF validation error."""
    pass


class PDFValidator:
    """Validates PDF files before processing.

    Validation is intentionally defensive: uploads are untrusted input even
    when they come from a local user, and a malformed PDF must never be able to
    exhaust memory or leave a partially processed document behind.
    """

    MAX_FILE_SIZE_MB = 50
    MIN_PDF_SIGNATURE = b'%PDF'

    # Active-content markers that have no purpose in a text-extraction
    # pipeline. Rejecting them prevents embedded scripts/launchers from ever
    # reaching a viewer that may execute them.
    ACTIVE_CONTENT_MARKERS = (
        b"/JavaScript",
        b"/Launch",
        b"/EmbeddedFile",
        b"/OpenAction",
    )

    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: Optional[int] = None) -> None:
        """Validate file size."""
        limit_mb = max_size_mb if max_size_mb is not None else PDFValidator.MAX_FILE_SIZE_MB
        try:
            size_bytes = os.path.getsize(file_path)
        except OSError:
            raise PDFValidationError("File could not be read")
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > limit_mb:
            raise PDFValidationError(f"File too large: {size_mb:.2f}MB (max {limit_mb}MB)")
        if size_bytes == 0:
            raise PDFValidationError("Empty file")

    @staticmethod
    def validate_pdf_signature(file_path: str) -> None:
        """Validate PDF file signature."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
        except OSError:
            raise PDFValidationError("File could not be read")
        if not header.startswith(PDFValidator.MIN_PDF_SIGNATURE):
            raise PDFValidationError("Invalid PDF file signature")

    @staticmethod
    def has_active_content(file_path: str) -> bool:
        """Scan for PDF JavaScript/launch/embedded-file markers.

        Scanning is chunked so a large file is not read into memory at once.
        A small overlap handles markers split across chunk boundaries.
        """
        try:
            with open(file_path, 'rb') as handle:
                overlap = max(len(marker) for marker in PDFValidator.ACTIVE_CONTENT_MARKERS) - 1
                previous = b""
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        return False
                    combined = previous + chunk
                    if any(marker in combined for marker in PDFValidator.ACTIVE_CONTENT_MARKERS):
                        return True
                    previous = combined[-overlap:] if overlap else b""
        except OSError:
            raise PDFValidationError("File could not be read")

    @staticmethod
    def validate_pdf_content(file_path: str, max_pages: Optional[int] = None,
                             allow_active_content: Optional[bool] = None) -> dict:
        """Validate PDF content and extract metadata."""
        page_limit = max_pages if max_pages is not None else settings.MAX_PDF_PAGES
        active_allowed = (
            allow_active_content
            if allow_active_content is not None
            else settings.ALLOW_PDF_ACTIVE_CONTENT
        )
        if not active_allowed and PDFValidator.has_active_content(file_path):
            raise PDFValidationError("PDF contains active content and is not supported")

        doc = None
        try:
            doc = fitz.open(file_path)
            if doc.needs_pass or doc.is_encrypted:
                raise PDFValidationError("Encrypted PDFs are not supported")
            if len(doc) == 0:
                raise PDFValidationError("PDF has no pages")
            if len(doc) > page_limit:
                raise PDFValidationError(
                    f"PDF has too many pages: {len(doc)} (max {page_limit})"
                )

            page_count = len(doc)
            # Check if scanned (little extractable text)
            total_text = ""
            for page_num in range(min(3, page_count)):  # Check first 3 pages
                page = doc[page_num]
                total_text += page.get_text()[:100_000]

            if len(total_text.strip()) < 50:
                raise PDFValidationError("Scanned PDF detected with little extractable text. OCR is not yet supported.")

            return {"page_count": page_count, "is_scannable": True}
        except PDFValidationError:
            raise
        except Exception as e:
            raise PDFValidationError(f"Corrupted or invalid PDF: {type(e).__name__}")
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    @staticmethod
    def validate_upload(file_path: str, content_type: str = None,
                        max_size_mb: Optional[int] = None,
                        max_pages: Optional[int] = None,
                        allow_active_content: Optional[bool] = None) -> dict:
        """Run all validations."""
        PDFValidator.validate_file_size(file_path, max_size_mb=max_size_mb)
        PDFValidator.validate_pdf_signature(file_path)
        return PDFValidator.validate_pdf_content(
            file_path,
            max_pages=max_pages,
            allow_active_content=allow_active_content,
        )
