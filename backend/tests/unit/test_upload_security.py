"""Upload validation, streaming limits, and failed-job cleanup tests."""
import io

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from app.api.v1 import documents
from app.ingestion.validator import PDFValidationError


@pytest.mark.unit
def test_upload_rejects_wrong_mime_type(client: TestClient):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("ok.pdf", b"%PDF-1.4 content", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid file type"


@pytest.mark.unit
def test_upload_rejects_wrong_extension(client: TestClient):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("ok.txt", b"%PDF-1.4 content", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid file extension"


@pytest.mark.unit
def test_upload_does_not_use_client_filename_for_storage(
    client: TestClient, tmp_path, monkeypatch
):
    monkeypatch.setattr(documents.settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        documents.PDFValidator,
        "validate_upload",
        lambda *args, **kwargs: {"page_count": 1},
    )
    monkeypatch.setattr(
        documents.PDFExtractor,
        "extract_text",
        lambda *args, **kwargs: [
            {"page_number": 1, "text": "safe extracted text", "char_count": 19}
        ],
    )
    created = {}
    monkeypatch.setattr(
        documents.DocumentDB,
        "create_document",
        lambda *args, **kwargs: created.setdefault("args", args),
    )
    monkeypatch.setattr(
        documents.DocumentDB, "update_document_status", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        documents.ChunkDB, "create_chunks_idempotent", lambda *args, **kwargs: 1
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "..\\..\\evil.pdf",
                b"%PDF-1.4\nsynthetic content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.pdf"
    # The stored path is a server-generated UUID, not the client name.
    assert created["args"][1] == "evil.pdf"
    stored = list(tmp_path.iterdir())
    assert len(stored) == 1
    assert stored[0].name.endswith(".pdf")
    assert stored[0].name != "evil.pdf"


@pytest.mark.unit
def test_failed_validation_removes_partial_file(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setattr(documents.settings, "UPLOAD_DIR", str(tmp_path))

    def reject(*args, **kwargs):
        raise PDFValidationError("Invalid PDF file signature")

    monkeypatch.setattr(documents.PDFValidator, "validate_upload", reject)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "Invalid PDF file signature" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_failed_processing_marks_failed_and_removes_file(
    client: TestClient, tmp_path, monkeypatch
):
    monkeypatch.setattr(documents.settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        documents.PDFValidator,
        "validate_upload",
        lambda *args, **kwargs: {"page_count": 1},
    )
    monkeypatch.setattr(
        documents.DocumentDB, "create_document", lambda *args, **kwargs: None
    )
    updates = []
    monkeypatch.setattr(
        documents.DocumentDB,
        "update_document_status",
        lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        documents.ChunkDB, "delete_chunks_for_document", lambda *args, **kwargs: None
    )

    def fail_extraction(*args, **kwargs):
        raise PDFValidationError("Document text exceeds the maximum allowed size")

    monkeypatch.setattr(documents.PDFExtractor, "extract_text", fail_extraction)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("big.pdf", b"%PDF-1.4\nsynthetic", "application/pdf")},
    )

    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []
    assert any(call[0][1] == "failed" for call in updates)


@pytest.mark.unit
async def test_streaming_upload_limit_stops_before_disk_exhaustion(tmp_path):
    upload = UploadFile(file=io.BytesIO(b"x" * 50), filename="big.pdf")
    destination = tmp_path / "upload.part"

    with pytest.raises(documents.UploadTooLarge):
        await documents._stream_upload_to_file(upload, destination, max_bytes=10)

    await upload.close()
    assert destination.stat().st_size <= 10
