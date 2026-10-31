"""Bounded traversal, parameterization, and document-scope tests."""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.config import settings
from app.db.graph import GraphDB
from app.db.graph_explore import GraphExplorer
from app.retrieval.graphrag import GraphRAGRetriever


class FakeSession:
    def __init__(self, recorder):
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.recorder.append((query, params))
        return []


class FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self, database=None):
        return FakeSession(self.calls)


@pytest.mark.unit
def test_neighbor_query_is_parameterized_and_bounded():
    explorer = GraphExplorer()
    driver = FakeDriver()
    explorer.connect = lambda: driver

    malicious = "x') MATCH (n) DETACH DELETE n //"
    explorer.get_entity_neighbors(
        malicious,
        depth=99,
        limit=999,
        rel_type=malicious,
        document_id="doc-1",
    )

    assert driver.calls
    query, params = driver.calls[0]
    assert "*1..3" in query
    assert "DETACH DELETE" not in query
    assert malicious not in query
    assert params["entity_id"] == malicious
    assert params["rel_type"] == malicious
    assert params["limit"] == 100
    assert params["doc_id"] == "doc-1"
    assert "r.document_id = $doc_id" in query


@pytest.mark.unit
def test_entity_search_does_not_interpolate_query():
    explorer = GraphExplorer()
    driver = FakeDriver()
    explorer.connect = lambda: driver

    payload = "'; MATCH (n) DETACH DELETE n //"
    explorer.search_entities(payload, document_id="doc-1", limit=999)

    query, params = driver.calls[0]
    assert payload not in query
    assert params["search_term"] == payload
    assert params["limit"] == 100


@pytest.mark.unit
def test_document_graph_view_requires_scoped_relationships():
    db = GraphDB()
    driver = FakeDriver()
    db.driver = driver

    db.get_graph(document_id="doc-1", limit=999)

    query, params = driver.calls[0]
    assert "(e2:Entity)-[:SOURCED_FROM]->(d)" in query
    assert "doc-1" not in query
    assert params["doc_id"] == "doc-1"
    assert params["limit"] == settings.MAX_GRAPH_LIMIT


@pytest.mark.unit
def test_graph_retrieval_document_scope():
    retriever = GraphRAGRetriever()
    driver = FakeDriver()
    retriever.graph_db.connect = lambda: driver

    retriever._graph_retrieval("question", document_id="doc-1")

    query, params = driver.calls[0]
    assert "(e2:Entity)-[:SOURCED_FROM]->(d)" in query
    assert "doc-1" not in query
    assert params["doc_id"] == "doc-1"
    assert f"*1..{retriever.graph_traversal_depth}" in query


@pytest.mark.unit
def test_graph_deletion_removes_document_relationship_evidence():
    queries = []

    class FakeTransaction:
        def run(self, query, **params):
            queries.append((query, params))

    GraphDB._delete_document(FakeTransaction(), "doc-1")

    assert len(queries) == 4
    joined = "\n".join(query for query, _ in queries)
    assert "doc-1" not in joined
    assert "r.document_id = $doc_id" in joined
    assert any(params.get("doc_id") == "doc-1" for _, params in queries)


@pytest.mark.unit
async def test_retrieval_context_is_bounded_even_with_huge_evidence():
    retriever = GraphRAGRetriever()
    retriever.max_context_tokens = 10  # 40-character budget
    retriever.vector_db = Mock()
    retriever.vector_db.search_similar = AsyncMock(return_value=[{
        "chunk_id": "c1",
        "document_id": "d1",
        "page_number": 1,
        "text": "X" * 100_000,
        "similarity": 0.9,
    }])
    retriever._graph_retrieval = Mock(return_value=[])

    result = await retriever.retrieve("question")

    assert result["has_evidence"] is True
    assert len(result["context"]) <= 40
    assert len(result["evidence"][0]["text"]) <= settings.MAX_EVIDENCE_CHARS


@pytest.mark.unit
def test_document_scoped_neighbors_keep_isolated_entities():
    """An entity with no relationships must not disappear behind the doc filter."""
    explorer = GraphExplorer()
    driver = FakeDriver()
    explorer.connect = lambda: driver

    explorer.get_entity_neighbors(
        "abc123", document_id="doc-1", node_type="Person", rel_type="USES"
    )

    query, params = driver.calls[0]
    assert "(neighbor IS NULL OR neighbor <> e)" in query
    assert "rel IS NULL OR (all(r in rel" in query
    assert "(neighbor IS NULL OR neighbor.type = $node_type)" in query
    assert "(rel IS NULL OR all(r in rel WHERE r.type = $rel_type))" in query
    assert params["node_type"] == "Person"
    assert params["rel_type"] == "USES"


@pytest.mark.unit
def test_document_graph_view_keeps_entities_without_relationships():
    db = GraphDB()
    driver = FakeDriver()
    db.driver = driver

    db.get_graph(document_id="doc-1")

    query, _ = driver.calls[0]
    assert "e2 IS NULL OR e <> e2" in query


@pytest.mark.unit
def test_entity_sourcing_relationship_records_document_id():
    calls = []

    class RecordingTx:
        def run(self, query, **params):
            calls.append((query, params))

    GraphDB._create_entity(
        RecordingTx(),
        "entity-id",
        "Alice",
        "Person",
        [],
        0.9,
        "chunk-1",
        "doc-1",
        2,
        "Alice works here",
    )

    query, params = calls[0]
    assert "r.document_id = $doc_id" in query
    assert params["doc_id"] == "doc-1"


@pytest.mark.unit
async def test_graph_extraction_route_times_out(monkeypatch):
    from fastapi import HTTPException

    import app.api.v1.graph as graph_api

    monkeypatch.setattr(
        graph_api.DocumentDB,
        "document_exists",
        staticmethod(lambda _id: True),
    )
    monkeypatch.setattr(
        graph_api.ChunkDB,
        "get_chunks_by_document",
        staticmethod(
            lambda _id, limit=None: [
                {
                    "chunk_id": "c1",
                    "document_id": _id,
                    "page_number": 1,
                    "chunk_text": "text",
                }
            ]
        ),
    )

    async def slow_extraction(_chunk):
        await asyncio.sleep(30)

    monkeypatch.setattr(
        graph_api.graph_extraction_service, "extract_from_chunk", slow_extraction
    )
    monkeypatch.setattr(graph_api.settings, "GRAPH_EXTRACTION_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(HTTPException) as excinfo:
        await graph_api.extract_graph("00000000-0000-4000-8000-000000000000")

    assert excinfo.value.status_code == 504
