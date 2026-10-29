#!/usr/bin/env python
r"""End-to-end smoke test for a running KnowledgeGraph AI backend.

This is a live test: it requires the backend, PostgreSQL+pgvector, Neo4j and
Ollama to be running. It is intentionally excluded from `pytest` collection
(the file name does not match test discovery) because it needs a live stack
and real model inference.

Usage (from backend/):

    .\.venv\Scripts\python.exe scripts\smoke.py --base-url http://127.0.0.1:8000

The test creates a temporary one-page PDF, uploads it, indexes it, asks a
question, extracts and reads the graph, deletes the document, and verifies
cleanup in the upload directory, PostgreSQL and Neo4j.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx
import psycopg
from neo4j import GraphDatabase

try:
    import pymupdf
except ImportError:  # older PyMuPDF releases expose only the fitz alias
    import fitz as pymupdf  # type: ignore

from app.core.config import settings

# Keep this below CHUNK_SIZE (512) so the fixture produces exactly one chunk.
# One chunk keeps the live test inside the per-request extraction timeout on
# CPU-only Ollama installs while still exercising every stage.
PAGE_TEXT = (
    "Ada Lovelace worked with Charles Babbage on the Analytical Engine in London. "
    "In 1843 she published extensive notes about the machine. Ada Lovelace is often "
    "called the first computer programmer. Her notes described how the engine could "
    "compute Bernoulli numbers. Charles Babbage designed the Analytical Engine and "
    "the Difference Engine. The Analytical Engine was a mechanical general-purpose computer."
)
PAGES = [PAGE_TEXT]

QUESTION = (
    "Who worked with Charles Babbage on the Analytical Engine, "
    "and what did she publish in 1843?"
)


class SmokeFailure(Exception):
    """Raised when a smoke-test assertion fails."""


def check(name: str, condition: bool, detail: str = "") -> None:
    label = "PASS" if condition else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{label}] {name}{suffix}", flush=True)
    if not condition:
        raise SmokeFailure(name)


def expect(response: httpx.Response, status_code: int = 200) -> dict:
    if response.status_code != status_code:
        body = response.text[:800]
        raise SmokeFailure(
            f"HTTP {response.status_code} for {response.request.method} "
            f"{response.request.url} (expected {status_code}): {body}"
        )
    if not response.content:
        return {}
    return response.json()


def make_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    y = 72
    for line in PAGE_TEXT.splitlines():
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    document.save(str(path))
    document.close()


def wait_for_ready(client: httpx.Client, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = client.get("/api/v1/health/ready", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("ready"):
                    return
                last_error = str(data.get("dependencies", {}))
        except Exception as exc:  # noqa: BLE001 - surfaced below
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise SmokeFailure(f"Backend did not become ready: {last_error}")


def upload_dir() -> Path:
    directory = Path(settings.UPLOAD_DIR)
    if not directory.is_absolute():
        # The documented startup command runs uvicorn from backend/.
        directory = (BACKEND_DIR / directory).resolve()
    return directory


def postgres_counts(document_id: str) -> tuple[int, int]:
    with psycopg.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=10,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM documents WHERE document_id = %s",
                (document_id,),
            )
            documents = cursor.fetchone()[0]
            cursor.execute(
                "SELECT count(*) FROM document_chunks WHERE document_id = %s",
                (document_id,),
            )
            chunks = cursor.fetchone()[0]
    return documents, chunks


def neo4j_counts(document_id: str) -> tuple[int, int]:
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    try:
        documents = driver.execute_query(
            "MATCH (d:Document {id: $document_id}) RETURN count(d) AS n",
            document_id=document_id,
            database_=settings.NEO4J_DATABASE,
        ).records[0]["n"]
        relationships = driver.execute_query(
            "MATCH ()-[r]->() WHERE r.document_id = $document_id "
            "RETURN count(r) AS n",
            document_id=document_id,
            database_=settings.NEO4J_DATABASE,
        ).records[0]["n"]
    finally:
        driver.close()
    return documents, relationships


def run(base_url: str, ready_timeout: float) -> None:
    timeout = httpx.Timeout(900.0, connect=10.0)
    document_id: str | None = None
    deleted = False

    client = httpx.Client(
        base_url=base_url, timeout=timeout, follow_redirects=True
    )
    try:
        print(f"Waiting for {base_url} ...", flush=True)
        wait_for_ready(client, ready_timeout)
        print("Backend is ready.", flush=True)

        with tempfile.TemporaryDirectory(prefix="kg-smoke-") as temp_dir:
            pdf_path = Path(temp_dir) / "knowledgegraph_smoke.pdf"
            make_pdf(pdf_path)
            print(f"Created test PDF: {pdf_path.stat().st_size} bytes", flush=True)

            with pdf_path.open("rb") as handle:
                response = client.post(
                    "/api/v1/documents/upload",
                    files={
                        "file": (
                            "knowledgegraph_smoke.pdf",
                            handle,
                            "application/pdf",
                        )
                    },
                )
            uploaded = expect(response)
            document_id = uploaded["document_id"]
            check(
                "upload and text extraction",
                uploaded.get("status") == "completed" and uploaded.get("document_id"),
                f"document_id={document_id} status={uploaded.get('status')}",
            )

            detail = expect(client.get(f"/api/v1/documents/{document_id}"))
            page_text = " ".join(page["text"] for page in detail.get("pages", []))
            check(
                "stored page snapshot",
                len(detail.get("pages", [])) == len(PAGES) and "Ada Lovelace" in page_text,
                f"pages={len(detail.get('pages', []))}",
            )

            started = time.monotonic()
            indexing = expect(client.post(f"/api/v1/documents/{document_id}/index"))
            index_status = indexing["status"]
            check(
                "vector indexing",
                index_status["total"] > 0
                and index_status["indexed"] == index_status["total"]
                and index_status["failed"] == 0,
                f"{index_status} in {time.monotonic() - started:.1f}s",
            )

            reported = expect(
                client.get(f"/api/v1/documents/{document_id}/index/status")
            )
            check(
                "index status endpoint",
                reported["indexed"] == index_status["total"],
                str(reported),
            )

            started = time.monotonic()
            chat = expect(
                client.post(
                    "/api/v1/chat/",
                    json={"question": QUESTION, "document_id": document_id},
                )
            )
            answer = chat.get("answer", "")
            check(
                "chat retrieval and generation",
                chat.get("has_evidence") is True
                and len(answer) > 20
                and len(chat.get("citations", [])) > 0
                and any(
                    citation.get("document_id") == document_id
                    for citation in chat.get("citations", [])
                ),
                f"{len(answer)} chars, {len(chat.get('citations', []))} citations "
                f"in {time.monotonic() - started:.1f}s",
            )
            check(
                "answer names Ada Lovelace",
                "lovelace" in answer.lower() or "ada" in answer.lower(),
                answer.replace("\n", " ")[:160],
            )

            started = time.monotonic()
            extraction = expect(
                client.post(f"/api/v1/graph/extract/{document_id}")
            )
            check(
                "graph extraction",
                extraction["processed_chunks"] > 0
                and extraction["entities_extracted"] > 0
                and extraction["relationships_extracted"] > 0,
                f"{extraction['entities_extracted']} entities, "
                f"{extraction['relationships_extracted']} relationships "
                f"in {time.monotonic() - started:.1f}s",
            )

            graph = expect(
                client.get("/api/v1/graph/", params={"document_id": document_id})
            )
            labels = [node["label"] for node in graph.get("nodes", [])]
            check(
                "graph read (document scoped)",
                graph.get("total_nodes", 0) > 0,
                f"{graph.get('total_nodes')} nodes, {graph.get('total_edges')} edges; "
                f"labels={labels[:8]}",
            )

            search = expect(
                client.get(
                    "/api/v1/graph/entities/search",
                    params={"q": "Ada", "document_id": document_id},
                )
            )
            check(
                "entity search",
                isinstance(search, list) and len(search) > 0,
                f"{len(search)} matches",
            )

            entity_id = search[0]["id"]
            entity = expect(
                client.get(
                    f"/api/v1/graph/entities/{entity_id}",
                    params={"document_id": document_id},
                )
            )
            check(
                "entity detail and neighbors",
                entity.get("id") == entity_id and len(entity.get("neighbors", [])) > 0,
                f"label={entity.get('label')} neighbors={len(entity.get('neighbors', []))}",
            )

            stored_file = upload_dir() / f"{document_id}.pdf"
            check("stored PDF exists before deletion", stored_file.is_file())

            deletion = expect(client.delete(f"/api/v1/documents/{document_id}"))
            deleted = True
            check(
                "delete reports complete cleanup",
                deletion.get("graph_deleted") is True
                and deletion.get("file_removed") is True
                and deletion.get("warnings") == [],
                str(deletion),
            )
            check("stored PDF removed", not stored_file.exists())

            check(
                "document is gone from the API",
                client.get(f"/api/v1/documents/{document_id}").status_code == 404,
            )
            documents, chunks = postgres_counts(document_id)
            check(
                "PostgreSQL cleanup",
                documents == 0 and chunks == 0,
                f"documents={documents} chunks={chunks}",
            )
            graph_documents, graph_relationships = neo4j_counts(document_id)
            check(
                "Neo4j cleanup",
                graph_documents == 0 and graph_relationships == 0,
                f"documents={graph_documents} relationships={graph_relationships}",
            )

        print("\nAll smoke checks passed.", flush=True)
    finally:
        if document_id and not deleted:
            print(
                f"Cleanup: deleting leftover test document {document_id} ...",
                flush=True,
            )
            try:
                client.delete(f"/api/v1/documents/{document_id}", timeout=60.0)
            except Exception as exc:  # noqa: BLE001 - best effort cleanup
                print(
                    f"Cleanup failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Running backend base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for /api/v1/health/ready (default: 60)",
    )
    args = parser.parse_args()

    try:
        run(args.base_url.rstrip("/"), args.ready_timeout)
    except SmokeFailure as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # noqa: BLE001 - keep the CLI failure readable
        print(
            f"\nSMOKE TEST ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
