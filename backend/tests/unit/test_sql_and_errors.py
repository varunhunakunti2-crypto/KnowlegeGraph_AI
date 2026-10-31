"""SQL/Cypher parameterization and error-response redaction tests."""
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import documents
from app.core.config import settings
from app.db.documents import DocumentDB
from app.db.vectors import VectorDB
from app.ingestion.validator import PDFValidationError

DOC_ID = "123e4567-e89b-12d3-a456-426614174000"


class FakeCursor:
    def __init__(self, recorder):
        self.recorder = recorder

    def execute(self, query, params=None):
        self.recorder.append((query, params))

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.recorder = []

    def cursor(self):
        return FakeCursor(self.recorder)

    def close(self):
        return None

    def commit(self):
        return None

    def rollback(self):
        return None


@pytest.mark.unit
def test_sql_uses_bound_parameters(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr("app.db.documents.psycopg.connect", lambda **kwargs: connection)

    payload = "'; DROP TABLE documents;--"
    assert DocumentDB.get_document(payload) is None

    query, params = connection.recorder[0]
    assert payload not in query
    assert params == (payload,)
    assert "%s" in query


@pytest.mark.unit
async def test_vector_search_parameterizes_and_clamps(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(VectorDB, "get_connection", lambda: connection)
    monkeypatch.setattr(
        "app.db.vectors.embedding_service.embed_text",
        AsyncMock(return_value=[0.1] * 768),
    )

    payload = "doc-1' OR '1'='1"
    results = await VectorDB.search_similar(payload, top_k=9999, document_id=payload)

    assert results == []
    query, params = connection.recorder[0]
    assert payload not in query
    assert payload in params
    assert params[-1] == settings.MAX_TOP_K
    assert "document_id = %s" in query


@pytest.mark.unit
def test_document_endpoint_hides_internal_paths(client: TestClient, monkeypatch):
    def fail(_):
        raise FileNotFoundError(r"C:\Users\alice\secret\database.py")

    monkeypatch.setattr(documents.DocumentDB, "get_document", fail)

    response = client.get(f"/api/v1/documents/{DOC_ID}")

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve document"
    assert "C:\\Users" not in response.text
    assert "database.py" not in response.text


@pytest.mark.unit
def test_upload_validation_error_redacts_paths(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setattr(documents.settings, "UPLOAD_DIR", str(tmp_path))

    def fail(*args, **kwargs):
        raise PDFValidationError(r"Corrupted C:\Users\alice\secret.pdf")

    monkeypatch.setattr(documents.PDFValidator, "validate_upload", fail)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 400
    assert "C:\\Users" not in response.json()["detail"]
    assert "[path]" in response.json()["detail"]


@pytest.mark.unit
def test_indexing_error_does_not_leak_exception_details(client: TestClient, monkeypatch):
    monkeypatch.setattr(documents.DocumentDB, "document_exists", lambda _: True)
    monkeypatch.setattr(
        documents.VectorDB,
        "index_document_chunks",
        AsyncMock(side_effect=RuntimeError("postgresql://user:supersecret@localhost/db")),
    )

    response = client.post(f"/api/v1/documents/{DOC_ID}/index")

    assert response.status_code == 500
    assert response.json()["detail"] == "Indexing failed"
    assert "supersecret" not in response.text
    assert "postgresql://" not in response.text


@pytest.mark.unit
def test_chat_rejects_malformed_document_id(client: TestClient):
    response = client.post(
        "/api/v1/chat/",
        json={"question": "hello", "document_id": "'; DROP TABLE documents;--"},
    )
    assert response.status_code == 400


@pytest.mark.unit
def test_chat_rejects_unknown_document(client: TestClient, monkeypatch):
    monkeypatch.setattr(documents.DocumentDB, "document_exists", lambda _: False)
    response = client.post(
        "/api/v1/chat/",
        json={"question": "hello", "document_id": DOC_ID},
    )
    assert response.status_code == 404


@pytest.mark.unit
def test_chat_question_length_is_bounded(client: TestClient):
    response = client.post(
        "/api/v1/chat/",
        json={"question": "x" * (settings.MAX_QUESTION_CHARS + 1)},
    )
    assert response.status_code == 422
