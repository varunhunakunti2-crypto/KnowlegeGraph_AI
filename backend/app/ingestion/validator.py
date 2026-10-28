"""PDF validation."""
import os
import fitz  # PyMuPDF


class PDFValidationError(Exception):
    """PDF validation error."""
    pass


class PDFValidator:
    """Validates PDF files before processing."""
    
    MAX_FILE_SIZE_MB = 50
    MIN_PDF_SIGNATURE = b'%PDF'
    
    @staticmethod
    def validate_file_size(file_path: str) -> None:
        """Validate file size."""
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > PDFValidator.MAX_FILE_SIZE_MB:
            raise PDFValidationError(f"File too large: {size_mb:.2f}MB (max {PDFValidator.MAX_FILE_SIZE_MB}MB)")
        if size_bytes == 0:
            raise PDFValidationError("Empty file")
    
    @staticmethod
    def validate_pdf_signature(file_path: str) -> None:
        """Validate PDF file signature."""
        with open(file_path, 'rb') as f:
            header = f.read(8)
        if not header.startswith(PDFValidator.MIN_PDF_SIGNATURE):
            raise PDFValidationError("Invalid PDF file signature")
    
    @staticmethod
    def validate_pdf_content(file_path: str) -> dict:
        """Validate PDF content and extract metadata."""
        try:
            doc = fitz.open(file_path)
            if doc.is_encrypted:
                doc.close()
                raise PDFValidationError("Encrypted PDFs are not supported")
            if len(doc) == 0:
                doc.close()
                raise PDFValidationError("PDF has no pages")
            
            page_count = len(doc)
            # Check if scanned (little extractable text)
            total_text = ""
            for page_num in range(min(3, page_count)):  # Check first 3 pages
                page = doc[page_num]
                total_text += page.get_text()
            
            doc.close()
            
            if len(total_text.strip()) < 50:
                raise PDFValidationError("Scanned PDF detected with little extractable text. OCR is not yet supported.")
            
            return {"page_count": page_count, "is_scannable": True}
        except PDFValidationError:
            raise
        except Exception as e:
            raise PDFValidationError(f"Corrupted or invalid PDF: {type(e).__name__}")
    
    @staticmethod
    def validate_upload(file_path: str, content_type: str = None) -> dict:
        """Run all validations."""
        PDFValidator.validate_file_size(file_path)
        PDFValidator.validate_pdf_signature(file_path)
        return PDFValidator.validate_pdf_content(file_path)
