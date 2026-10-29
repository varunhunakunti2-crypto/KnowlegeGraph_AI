"""Server contract consumed by the frontend (step 11, server side).

The browser test in ``frontend/tests/e2e`` renders these same payload shapes.
These tests fail if the API stops returning a field the UI reads, even when the
real databases/services are unavailable.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.api.v1 import chat as chat_api
from app.api.v1 import graph as graph_api

pytestmark = pytest.mark.evaluation

DOCUMENT_ID = "123e4567-e89b-12d3-a456-426614174000"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _document():
    return {
        "document_id": DOCUMENT_ID,
        "original_filename": "ada_lovelace.pdf",
        "file_size": 1024,
        "page_count": 3,
        "status": "completed",
        "uploaded_at": "2026-01-01T00:00:00",
        "processed_at": "2026-01-01T00:01:00",
        "error_message": None,
    }


def _retrieval_result():
    return {
        "evidence": [
            {
                "source_type": "vector",
                "chunk_id": "chunk-ada-1",
                "document_id": DOCUMENT_ID,
                "page_number": 1,
                "text": "Charles Babbage designed the Analytical Engine in 1837.",
                "score": 0.55,
                "entity_id": None,
                "relationship_info": None,
            },
            {
                "source_type": "graph",
                "chunk_id": "chunk-ada-2",
                "document_id": DOCUMENT_ID,
                "page_number": 2,
                "text": "Ada Lovelace worked with Charles Babbage.",
                "score": 0.34,
                "entity_id": "entity-1",
                "relationship_info": {"predicate": "RELATED_TO"},
            },
        ],
        "has_evidence": True,
        "context": "…",
        "total_tokens_estimate": 25,
    }


def _generation_result():
    return {
        "answer": "Charles Babbage designed the Analytical Engine in 1837. [chunk:chunk-ada-1]",
        "citations": [
            {
                "chunk_id": "chunk-ada-1",
                "document_id": DOCUMENT_ID,
                "page_number": 1,
                "snippet": "Charles Babbage designed the Analytical Engine in 1837.",
            }
        ],
        "evidence_snippets": [
            {
                "chunk_id": "chunk-ada-1",
                "document_id": DOCUMENT_ID,
                "page_number": 1,
                "text": "Charles Babbage designed the Analytical Engine in 1837.",
            }
        ],
    }


@pytest.mark.evaluation
def test_chat_response_has_every_field_the_ui_renders(client, monkeypatch):
    monkeypatch.setattr(
        chat_api.graphrag_retriever,
        "retrieve",
        AsyncMock(return_value=_retrieval_result()),
    )
    monkeypatch.setattr(
        chat_api.answer_generator,
        "generate_answer",
        AsyncMock(return_value=_generation_result()),
    )
    monkeypatch.setattr(chat_api.DocumentDB, "get_document", lambda _document_id: _document())

    response = client.post(
        "/api/v1/chat/",
        json={"question": "Who designed the Analytical Engine?", "document_id": DOCUMENT_ID},
    )
    assert response.status_code == 200
    body = response.json()

    # frontend/types/api.ts -> ChatResponse
    assert isinstance(body["question"], str)
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["citations"], list) and body["citations"]
    assert isinstance(body["evidence"], list)
    assert body["has_evidence"] is True
    assert isinstance(body["retrieval_metadata"], dict)
    assert body["retrieval_metadata"]["method"] == "graphrag"
    assert body["retrieval_metadata"]["evidence_count"] == 2
    assert body["retrieval_metadata"]["vector_results"] == 1
    assert body["retrieval_metadata"]["graph_results"] == 1

    citation = body["citations"][0]
    # Fields rendered by frontend/app/chat/page.tsx.
    assert citation["chunk_id"] == "chunk-ada-1"
    assert citation["document_id"] == DOCUMENT_ID
    assert citation["document_name"] == "ada_lovelace.pdf"
    assert citation["page_number"] == 1
    assert citation["snippet"]


def test_chat_without_evidence_reports_abstention(client, monkeypatch):
    monkeypatch.setattr(
        chat_api.graphrag_retriever,
        "retrieve",
        AsyncMock(
            return_value={
                "evidence": [],
                "has_evidence": False,
                "context": "",
                "total_tokens_estimate": 0,
                "message": "Insufficient evidence to answer the question",
            }
        ),
    )
    generate = AsyncMock()
    monkeypatch.setattr(chat_api.answer_generator, "generate_answer", generate)

    response = client.post("/api/v1/chat/", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    body = response.json()
    assert body["has_evidence"] is False
    assert body["citations"] == []
    assert body["message"] == "Insufficient evidence"
    generate.assert_not_awaited()


def test_graph_response_shape_matches_react_flow_usage(client, monkeypatch):
    nodes = [
        {"id": "e1", "label": "Ada Lovelace", "type": "Person", "aliases": []},
        {"id": "e2", "label": "Analytical Engine", "type": "Technology", "aliases": []},
    ]
    edges = [
        {
            "id": "e1-e2",
            "source": "e1",
            "target": "e2",
            "label": "RELATED_TO",
            "confidence": 0.9,
            "evidence": [],
        }
    ]
    monkeypatch.setattr(
        graph_api.graph_explorer,
        "get_document_graph",
        lambda _document_id, limit_nodes=100: {"nodes": nodes, "edges": edges},
    )

    response = client.get(f"/api/v1/graph/?document_id={DOCUMENT_ID}")
    assert response.status_code == 200
    body = response.json()

    # frontend/app/graph/page.tsx maps exactly these fields into React Flow.
    assert body["total_nodes"] == 2
    assert body["total_edges"] == 1
    assert {"id", "label", "type"} <= set(body["nodes"][0])
    assert {"id", "source", "target", "label"} <= set(body["edges"][0])


def test_typescript_contract_declares_rendered_fields():
    """Guard against the TypeScript types drifting away from the API tests."""

    types_file = REPO_ROOT / "frontend" / "types" / "api.ts"
    assert types_file.exists(), "frontend/types/api.ts is missing"
    text = types_file.read_text(encoding="utf-8")

    for expected in (
        "export interface ChatResponse",
        "answer: string",
        "citations: Citation[]",
        "has_evidence: boolean",
        "retrieval_metadata: Record<string, unknown>",
        "document_name?: string",
        "page_number: number",
        "snippet: string",
    ):
        assert expected in text, f"missing TypeScript contract field: {expected}"
