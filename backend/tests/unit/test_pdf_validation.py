"""Tests for PDF validation."""
import pytest
import os
import tempfile
from app.ingestion.validator import PDFValidator, PDFValidationError


@pytest.mark.unit
def test_empty_file_validation():
    """Test empty file is rejected."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
        f.write(b'')
        temp_path = f.name
    
    try:
        with pytest.raises(PDFValidationError, match="Empty file"):
            PDFValidator.validate_file_size(temp_path)
    finally:
        os.unlink(temp_path)


@pytest.mark.unit
def test_invalid_pdf_signature():
    """Test invalid PDF signature is rejected."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
        f.write(b'not a pdf file')
        temp_path = f.name
    
    try:
        with pytest.raises(PDFValidationError, match="Invalid PDF file signature"):
            PDFValidator.validate_pdf_signature(temp_path)
    finally:
        os.unlink(temp_path)


@pytest.mark.unit
def test_oversized_file_validation():
    """Test oversized file is rejected."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
        f.write(b'%PDF-1.4' + b'x' * (51 * 1024 * 1024))
        temp_path = f.name
    
    try:
        with pytest.raises(PDFValidationError, match="File too large"):
            PDFValidator.validate_file_size(temp_path)
    finally:
        os.unlink(temp_path)
