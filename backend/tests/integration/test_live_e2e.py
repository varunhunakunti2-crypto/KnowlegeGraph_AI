"""Live end-to-end test against the real stack and real local models.

Disabled by default. Run it explicitly:

    $env:KG_LIVE = "1"
    .\\.venv\\Scripts\\python.exe -m pytest tests/integration/test_live_e2e.py -v -s

When enabled, the test fails (rather than skips) if the backend, PostgreSQL,
Neo4j, or Ollama are unavailable. It is slow on CPU-only machines because the
real qwen3:4b model performs graph extraction and answer generation. A second
test uses a controlled graph payload so real Neo4j edge storage is covered even
when the model extracts no usable relationship.

Environment:
    KG_BASE_URL      backend base URL (default http://127.0.0.1:8000)
    KG_FRONTEND_URL  frontend URL for a serve check (default http://127.0.0.1:3000)
    KG_LIVE          must be "1" to enable this module
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import httpx
import pytest

from app.db.graph import graph_db
from app.schemas.graph import Entity, GraphExtractionResult, Relationship
from scripts.smoke import (
    expect,
    neo4j_counts,
    postgres_counts,
    upload_dir,
    wait_for_ready,
)
from tests.evaluation.pdfs import build_pdf_bytes

LIVE = os.environ.get("KG_LIVE") == "1"
BASE_URL = os.environ.get("KG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
FRONTEND_URL = os.environ.get("KG_FRONTEND_URL", "http://127.0.0.1:3000").rstrip("/")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.slow,
    pytest.mark.skipif(
        not LIVE,
        reason="Live stack test. Set KG_LIVE=1 to run against a running backend.",
    ),
]


def _fixture_text(token: str) -> str:
    return (
        f"In 2001, Mara Voss created the Aurora Institute {token} in Reykjavik. "
        f"The institute studies the northern lights. "
        f"Mara Voss leads the Aurora Institute {token}."
    )


def _server_rss_mb() -> float | None:
    """Best-effort RSS of the process listening on the backend port.

    Requires psutil; returns None when it is not installed so the test never
    fails just because an optional measurement dependency is missing.
    """

    try:
        import psutil  # type: ignore
    except Exception:
        return None

    port = 8000
    if ":" in BASE_URL.rsplit("/", 1)[-1]:
        try:
            port = int(BASE_URL.rsplit(":", 1)[-1])
        except ValueError:
            pass
    for connection in psutil.net_connections(kind="tcp"):
        if connection.laddr and connection.laddr.port == port and connection.status == "LISTEN":
            if connection.pid:
                try:
                    return psutil.Process(connection.pid).memory_info().rss / (1024 * 1024)
                except Exception:
                    return None
    return None


def test_live_upload_index_chat_graph_delete(tmp_path: Path, record_property):
    timeout = httpx.Timeout(900.0, connect=10.0)
    document_id: str | None = None
    deleted = False
    timings: dict[str, float] = {}
    rss_samples: dict[str, float] = {}

    client = httpx.Client(base_url=BASE_URL, timeout=timeout, follow_redirects=True)
    try:
        started = time.monotonic()
        wait_for_ready(client, timeout=60.0)
        timings["ready"] = time.monotonic() - started

        token = uuid.uuid4().hex[:6].upper()
        page_text = _fixture_text(token)
        pdf_path = tmp_path / f"live_e2e_{token}.pdf"
        pdf_path.write_bytes(build_pdf_bytes([page_text]))

        # 1-2. Upload and page-level extraction.
        started = time.monotonic()
        with pdf_path.open("rb") as handle:
            uploaded = expect(
                client.post(
                    "/api/v1/documents/upload",
                    files={"file": (pdf_path.name, handle, "application/pdf")},
                )
            )
        timings["upload"] = time.monotonic() - started
        document_id = uploaded["document_id"]
        assert uploaded["status"] == "completed"

        detail = expect(client.get(f"/api/v1/documents/{document_id}"))
        assert detail["page_count"] == 1
        assert len(detail["pages"]) == 1
        assert token in detail["pages"][0]["text"]

        # 3-4. Chunking + embeddings through the real index endpoint.
        started = time.monotonic()
        indexing = expect(client.post(f"/api/v1/documents/{document_id}/index"))
        timings["index"] = time.monotonic() - started
        index_status = indexing["status"]
        assert index_status["total"] > 0
        assert index_status["indexed"] == index_status["total"]
        assert index_status["failed"] == 0

        # 5-6. Real entity/relationship extraction into Neo4j.
        started = time.monotonic()
        extraction = expect(client.post(f"/api/v1/graph/extract/{document_id}"))
        timings["graph_extract"] = time.monotonic() - started
        assert extraction["processed_chunks"] > 0
        assert extraction["entities_extracted"] > 0
        # The small local model may extract an entity but omit relationships
        # (or name a relationship endpoint it never extracted). The round-trip
        # test below verifies edge storage independently of model variance.
        rss_samples["after_extract"] = _server_rss_mb() or 0.0

        # 7-10. Vector + graph retrieval, combined evidence, cited answer.
        started = time.monotonic()
        chat = expect(
            client.post(
                "/api/v1/chat/",
                json={
                    "question": f"Who created the Aurora Institute {token}?",
                    "document_id": document_id,
                },
            )
        )
        timings["chat"] = time.monotonic() - started
        rss_samples["after_chat"] = _server_rss_mb() or 0.0

        answer = chat["answer"]
        assert chat["has_evidence"] is True
        assert len(answer) > 10
        assert chat["citations"], "real generation returned no citations"
        assert all(
            citation["document_id"] == document_id for citation in chat["citations"]
        )
        assert all(citation["page_number"] == 1 for citation in chat["citations"])
        assert "voss" in answer.lower() or "mara" in answer.lower(), answer[:300]

        graph = expect(
            client.get("/api/v1/graph/", params={"document_id": document_id})
        )
        assert graph["total_nodes"] > 0
        # Edge creation depends on the model listing both ends of a relationship
        # as entities. When it reports a usable relationship, the edge must exist.
        if extraction["relationships_extracted"]:
            assert graph["total_edges"] > 0
        assert all(node["label"] for node in graph["nodes"])

        search = expect(
            client.get(
                "/api/v1/graph/entities/search",
                params={"q": graph["nodes"][0]["label"], "document_id": document_id},
            )
        )
        assert search, "entity search returned no matches for an extracted node"
        entity = expect(
            client.get(
                f"/api/v1/graph/entities/{search[0]['id']}",
                params={"document_id": document_id},
            )
        )
        assert entity["id"] == search[0]["id"]
        assert entity["evidence"]

        # 11. Frontend serve check. Full browser rendering is covered by
        # frontend/tests/e2e (Playwright); this only proves the app is reachable.
        try:
            frontend = httpx.get(FRONTEND_URL, timeout=20.0, follow_redirects=True)
            frontend_ok = frontend.status_code == 200 and "KnowledgeGraph" in frontend.text
        except Exception:
            frontend_ok = False
        record_property("frontend_served", str(frontend_ok))
        print(f"\nfrontend served at {FRONTEND_URL}: {frontend_ok}")

        # 12. Deletion and cross-store cleanup.
        stored_file = upload_dir() / f"{document_id}.pdf"
        assert stored_file.is_file()
        started = time.monotonic()
        deletion = expect(client.delete(f"/api/v1/documents/{document_id}"))
        timings["delete"] = time.monotonic() - started
        deleted = True
        assert deletion["graph_deleted"] is True
        assert deletion["file_removed"] is True
        assert deletion["warnings"] == []
        assert not stored_file.exists()
        assert client.get(f"/api/v1/documents/{document_id}").status_code == 404

        documents, chunks = postgres_counts(document_id)
        assert documents == 0 and chunks == 0
        graph_documents, graph_relationships = neo4j_counts(document_id)
        assert graph_documents == 0 and graph_relationships == 0

        print("\nlive latency (seconds):")
        for stage, seconds in timings.items():
            print(f"  {stage:<14} {seconds:>8.2f}")
        print(
            f"extraction: {extraction['entities_extracted']} entities, "
            f"{extraction['relationships_extracted']} relationships"
        )
        print(f"graph: {graph['total_nodes']} nodes, {graph['total_edges']} edges")
        print(f"answers: {answer.replace(chr(10), ' ')[:240]}")
        print(
            "server RSS (MB, optional): "
            + json.dumps({key: round(value, 1) for key, value in rss_samples.items()})
        )
        record_property("live_latencies_s", json.dumps(timings))
        record_property("live_rss_mb", json.dumps(rss_samples))
    finally:
        if document_id and not deleted:
            print(f"\nCleanup: removing leftover live-test document {document_id}")
            try:
                client.delete(f"/api/v1/documents/{document_id}", timeout=120.0)
            except Exception as exc:  # noqa: BLE001 - best effort
                print(f"Cleanup failed: {type(exc).__name__}: {exc}")
        client.close()


def test_live_graph_storage_and_api_roundtrip():
    """Verify real Neo4j edge storage and graph reads without model variance.

    Model extraction is allowed to return no usable edges, so this test builds a
    controlled extraction result and checks that the real Neo4j layer stores
    edges, the graph API reads them back, and deletion removes them. It closes
    the gap between the in-memory deterministic suite and a live Neo4j store.
    """

    document_id = str(uuid.uuid4())
    suffix = uuid.uuid4().hex[:6]
    person = f"Mara Voss {suffix}"
    organization = f"Aurora Institute {suffix}"
    city = f"Reykjavik {suffix}"
    chunk_id = f"{document_id}-0"
    evidence = f"In 2001, {person} created the {organization} in {city}."

    extraction = GraphExtractionResult(
        entities=[
            Entity(
                name=person,
                type="Person",
                confidence=0.9,
                source_chunk_id=chunk_id,
                source_document_id=document_id,
                page_number=1,
                evidence=evidence,
            ),
            Entity(
                name=organization,
                type="Organization",
                confidence=0.9,
                source_chunk_id=chunk_id,
                source_document_id=document_id,
                page_number=1,
                evidence=evidence,
            ),
            Entity(
                name=city,
                type="Location",
                confidence=0.9,
                source_chunk_id=chunk_id,
                source_document_id=document_id,
                page_number=1,
                evidence=evidence,
            ),
        ],
        relationships=[
            Relationship(
                subject=person,
                predicate="CREATED",
                object=organization,
                confidence=0.9,
                source_chunk_id=chunk_id,
                source_document_id=document_id,
                page_number=1,
                evidence=evidence,
            ),
            Relationship(
                subject=organization,
                predicate="LOCATED_IN",
                object=city,
                confidence=0.9,
                source_chunk_id=chunk_id,
                source_document_id=document_id,
                page_number=1,
                evidence=evidence,
            ),
        ],
    )

    try:
        stored = graph_db.store_extraction(document_id, extraction)
        assert stored == {"entities_stored": 3, "relationships_stored": 2}

        graph = graph_db.get_graph(document_id, limit=10)
        assert {node["label"] for node in graph["nodes"]} == {
            person,
            organization,
            city,
        }
        assert {edge["label"] for edge in graph["edges"]} == {
            "CREATED",
            "LOCATED_IN",
        }
        # Outgoing-only traversal must return one edge per stored relationship,
        # not a forward and reverse copy.
        assert len(graph["edges"]) == 2

        client = httpx.Client(
            base_url=BASE_URL, timeout=httpx.Timeout(30.0, connect=10.0)
        )
        try:
            api_graph = expect(
                client.get("/api/v1/graph/", params={"document_id": document_id})
            )
        finally:
            client.close()
        assert {node["label"] for node in api_graph["nodes"]} == {
            person,
            organization,
            city,
        }
        assert {edge["label"] for edge in api_graph["edges"]} == {
            "CREATED",
            "LOCATED_IN",
        }
        assert len(api_graph["edges"]) == 2
    finally:
        graph_db.delete_document_graph(document_id)

    assert graph_db.get_graph(document_id, limit=10) == {"nodes": [], "edges": []}
