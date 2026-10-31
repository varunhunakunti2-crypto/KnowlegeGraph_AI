"""Resumable, batch-committed indexing behavior."""

import pytest

from app.core.memory import InsufficientMemoryError
from app.db.vectors import VectorDB


class FakeTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        if "SELECT chunk_id, chunk_text, chunk_index" in normalized:
            _document_id, last_index, limit = params
            self._rows = [
                row for row in self.connection.rows if row[2] > last_index
            ][:limit]
        elif "embedding_status = 'indexed'" in normalized:
            self.connection.indexed_updates.append(params[2])

    def executemany(self, query, params):
        self.connection.failed_updates.extend(item[0] for item in params)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.autocommit = False
        self.indexed_updates = []
        self.failed_updates = []
        self.closed = False

    def transaction(self):
        return FakeTransaction()

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


@pytest.mark.unit
async def test_indexing_commits_successful_batches_before_a_failure(monkeypatch):
    rows = [("c1", "one", 0), ("c2", "two", 1), ("c3", "three", 2)]
    connection = FakeConnection(rows)

    monkeypatch.setattr(VectorDB, "get_connection", staticmethod(lambda: connection))
    monkeypatch.setattr(
        VectorDB,
        "get_indexing_status",
        staticmethod(lambda _doc: {
            "total": 3, "indexed": 2, "pending": 0, "failed": 1,
            "percent_complete": 66.7,
        }),
    )
    monkeypatch.setattr("app.db.vectors.settings.INDEX_BATCH_SIZE", 2)
    monkeypatch.setattr("app.db.vectors.settings.BATCH_SIZE", 16)

    calls = {"count": 0}

    async def embed(texts):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("temporary model failure")
        return [[0.1] * 768 for _ in texts]

    monkeypatch.setattr("app.db.vectors.embedding_service.embed_batch", embed)

    result = await VectorDB.index_document_chunks("doc-1")

    # The first batch was applied and committed; the failed batch was marked
    # failed for a later retry. A retry starts after the last committed row.
    assert result["indexed"] == 2
    assert result["failed"] == 1
    assert result["status"] == "partial"
    assert result["resumable"] is True
    assert connection.indexed_updates == ["c1", "c2"]
    assert connection.failed_updates == ["c3"]
    assert connection.closed is True


@pytest.mark.unit
async def test_indexing_stops_and_reports_progress_on_out_of_memory(monkeypatch):
    rows = [("c1", "one", 0), ("c2", "two", 1)]
    connection = FakeConnection(rows)

    monkeypatch.setattr(VectorDB, "get_connection", staticmethod(lambda: connection))
    monkeypatch.setattr(
        VectorDB,
        "get_indexing_status",
        staticmethod(lambda _doc: {
            "total": 2, "indexed": 0, "pending": 0, "failed": 2,
            "percent_complete": 0.0,
        }),
    )
    monkeypatch.setattr("app.db.vectors.settings.INDEX_BATCH_SIZE", 2)
    monkeypatch.setattr("app.db.vectors.settings.BATCH_SIZE", 16)

    async def embed(_texts):
        raise InsufficientMemoryError("out of memory")

    monkeypatch.setattr("app.db.vectors.embedding_service.embed_batch", embed)

    with pytest.raises(InsufficientMemoryError) as excinfo:
        await VectorDB.index_document_chunks("doc-1")

    assert "resumable" in str(excinfo.value).lower()
    assert connection.failed_updates == ["c1", "c2"]
