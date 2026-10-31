"""Deletion consistency across file, SQL rows/embeddings, and graph evidence."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import documents

DOC_ID = "123e4567-e89b-12d3-a456-426614174000"


def _document():
    return {
        "document_id": DOC_ID,
        "original_filename": "test.pdf",
        "file_size": 10,
        "page_count": 1,
        "status": "completed",
        "uploaded_at": "2026-01-01T00:00:00",
        "processed_at": None,
        "error_message": None,
        "pages": [],
    }


@pytest.mark.unit
def test_delete_document_cleans_file_sql_and_graph(client: TestClient, tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    stored = upload_dir / f"{DOC_ID}.pdf"
    stored.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(documents.settings, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(documents.DocumentDB, "document_exists", lambda _: True)
    delete_row = Mock(return_value={"file_path": str(stored)})
    monkeypatch.setattr(documents.DocumentDB, "delete_document", delete_row)
    delete_graph = Mock()
    monkeypatch.setattr(documents.graph_db, "delete_document_graph", delete_graph)

    response = client.delete(f"/api/v1/documents/{DOC_ID}")

    assert response.status_code == 200
    assert response.json()["graph_deleted"] is True
    assert response.json()["file_removed"] is True
    delete_graph.assert_called_once_with(DOC_ID)
    delete_row.assert_called_once_with(DOC_ID)
    assert not stored.exists()


@pytest.mark.unit
def test_delete_succeeds_with_warning_when_graph_is_unavailable(
    client: TestClient, tmp_path, monkeypatch
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    stored = upload_dir / f"{DOC_ID}.pdf"
    stored.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(documents.settings, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(documents.DocumentDB, "document_exists", lambda _: True)
    monkeypatch.setattr(
        documents.DocumentDB,
        "delete_document",
        Mock(return_value={"file_path": str(stored)}),
    )
    monkeypatch.setattr(
        documents.graph_db,
        "delete_document_graph",
        Mock(side_effect=RuntimeError("neo4j unavailable")),
    )

    response = client.delete(f"/api/v1/documents/{DOC_ID}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["graph_deleted"] is False
    assert any("Graph" in warning for warning in payload["warnings"])
    assert not stored.exists()


@pytest.mark.unit
def test_delete_never_removes_files_outside_upload_dir(
    client: TestClient, tmp_path, monkeypatch
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    outside = tmp_path / "important.txt"
    outside.write_text("do not delete")

    monkeypatch.setattr(documents.settings, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(documents.DocumentDB, "document_exists", lambda _: True)
    monkeypatch.setattr(
        documents.DocumentDB,
        "delete_document",
        Mock(return_value={"file_path": str(outside)}),
    )
    monkeypatch.setattr(
        documents.graph_db, "delete_document_graph", Mock()
    )

    response = client.delete(f"/api/v1/documents/{DOC_ID}")

    assert response.status_code == 200
    assert response.json()["file_removed"] is False
    assert outside.exists()


@pytest.mark.unit
def test_delete_rejects_malformed_document_id(client: TestClient, monkeypatch):
    delete_graph = Mock()
    monkeypatch.setattr(documents.graph_db, "delete_document_graph", delete_graph)

    response = client.delete("/api/v1/documents/'; DROP TABLE documents;--")

    assert response.status_code == 400
    delete_graph.assert_not_called()


@pytest.mark.unit
def test_delete_missing_document_returns_404(client: TestClient, monkeypatch):
    monkeypatch.setattr(documents.DocumentDB, "document_exists", lambda _: False)
    delete_graph = Mock()
    monkeypatch.setattr(documents.graph_db, "delete_document_graph", delete_graph)

    response = client.delete(f"/api/v1/documents/{DOC_ID}")

    assert response.status_code == 404
    delete_graph.assert_not_called()
