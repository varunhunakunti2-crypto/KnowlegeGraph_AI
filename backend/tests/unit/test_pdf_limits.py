"""Tests for PDF resource and malicious-content limits."""
import pytest
import fitz

from app.ingestion.extractor import PDFExtractor
from app.ingestion.validator import PDFValidator, PDFValidationError


def _make_pdf(path, pages=2, text="This is a sufficiently long test document. " * 5):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


@pytest.mark.unit
def test_page_count_limit_is_enforced(tmp_path):
    pdf_path = tmp_path / "many-pages.pdf"
    _make_pdf(pdf_path, pages=3)
    with pytest.raises(PDFValidationError, match="too many pages"):
        PDFValidator.validate_pdf_content(str(pdf_path), max_pages=2)


@pytest.mark.unit
def test_active_content_is_rejected(tmp_path):
    pdf_path = tmp_path / "active.pdf"
    pdf_path.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /OpenAction << /S /JavaScript /JS "
        b"(app.alert('x')) >> >>\nendobj\n%%EOF\n"
    )
    assert PDFValidator.has_active_content(str(pdf_path)) is True
    with pytest.raises(PDFValidationError, match="active content"):
        PDFValidator.validate_pdf_content(
            str(pdf_path), allow_active_content=False
        )


@pytest.mark.unit
def test_extraction_page_limit_is_enforced(tmp_path):
    pdf_path = tmp_path / "pages.pdf"
    _make_pdf(pdf_path, pages=3)
    with pytest.raises(PDFValidationError, match="too many pages"):
        PDFExtractor.extract_text(str(pdf_path), max_pages=2)


@pytest.mark.unit
def test_extraction_character_limit_is_enforced(tmp_path):
    pdf_path = tmp_path / "long.pdf"
    _make_pdf(pdf_path, pages=1)
    with pytest.raises(PDFValidationError, match="maximum allowed size"):
        PDFExtractor.extract_text(str(pdf_path), max_chars=10)


@pytest.mark.unit
def test_validation_size_limit_can_be_overridden(tmp_path):
    pdf_path = tmp_path / "small.pdf"
    _make_pdf(pdf_path, pages=1)
    with pytest.raises(PDFValidationError, match="File too large"):
        PDFValidator.validate_file_size(str(pdf_path), max_size_mb=0)
