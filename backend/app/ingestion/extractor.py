"""PDF text extraction."""
import fitz  # PyMuPDF
from typing import List, Dict
from app.ingestion.validator import PDFValidationError


class PDFExtractor:
    """Extracts text from PDF files page by page."""
    
    @staticmethod
    def extract_text(file_path: str) -> List[Dict]:
        """
        Extract text page by page preserving page numbers.
        """
        try:
            doc = fitz.open(file_path)
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text)
                })
            doc.close()
            return pages
        except Exception as e:
            raise PDFValidationError(f"Failed to extract text: {type(e).__name__}")
