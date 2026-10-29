"""Tests for retrying graph deletion."""
from unittest.mock import MagicMock

import pytest

from app.db.graph import GraphDB


@pytest.mark.unit
def test_delete_document_graph_retries_transient_failure(monkeypatch):
    db = GraphDB()
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.execute_write.side_effect = [RuntimeError("transient"), None]

    monkeypatch.setattr(db, "connect", lambda: driver)
    monkeypatch.setattr("app.db.graph.time.sleep", lambda _: None)

    db.delete_document_graph("doc-1", retries=1)

    assert session.execute_write.call_count == 2
    session.execute_write.assert_called_with(db._delete_document, "doc-1")


@pytest.mark.unit
def test_delete_document_graph_raises_after_retries(monkeypatch):
    db = GraphDB()
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.execute_write.side_effect = RuntimeError("down")

    monkeypatch.setattr(db, "connect", lambda: driver)
    monkeypatch.setattr("app.db.graph.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError):
        db.delete_document_graph("doc-1", retries=2)

    assert session.execute_write.call_count == 3
