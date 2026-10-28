"""PDF text extraction."""
from typing import List, Dict, Optional

import fitz  # PyMuPDF

from app.core.config import settings
from app.ingestion.validator import PDFValidationError


class PDFExtractor:
    """Extracts text from PDF files page by page."""

    @staticmethod
    def extract_text(file_path: str, max_pages: Optional[int] = None,
                     max_chars: Optional[int] = None) -> List[Dict]:
        """
        Extract text page by page preserving page numbers.

        Page and character limits are enforced again here (in addition to
        validation) so a caller that skips validation still cannot cause
        unbounded memory use.
        """
        page_limit = max_pages if max_pages is not None else settings.MAX_PDF_PAGES
        char_limit = max_chars if max_chars is not None else settings.MAX_EXTRACTED_TEXT_CHARS
        doc = None
        try:
            doc = fitz.open(file_path)
            if len(doc) > page_limit:
                raise PDFValidationError(
                    f"PDF has too many pages: {len(doc)} (max {page_limit})"
                )
            pages = []
            total_chars = 0
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                total_chars += len(text)
                if total_chars > char_limit:
                    raise PDFValidationError(
                        "Document text exceeds the maximum allowed size"
                    )
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text)
                })
            return pages
        except PDFValidationError:
            raise
        except Exception as e:
            raise PDFValidationError(f"Failed to extract text: {type(e).__name__}")
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass
