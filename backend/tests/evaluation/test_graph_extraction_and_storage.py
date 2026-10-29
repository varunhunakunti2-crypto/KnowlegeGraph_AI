"""Structured graph extraction, sanitisation, storage, and deletion (steps 5-6)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.graph import APPROVED_ENTITY_TYPES, APPROVED_PREDICATES
from tests.evaluation.corpus import CORPUS
from tests.evaluation.pipeline import DeterministicPipeline

pytestmark = pytest.mark.evaluation


async def _pipeline(tmp_path: Path) -> DeterministicPipeline:
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()
    return pipeline


@pytest.mark.asyncio
async def test_extraction_is_sanitised_and_stored_with_provenance(tmp_path: Path):
    pipeline = await _pipeline(tmp_path)

    for stats in pipeline.ingest_stats.values():
        assert stats["entities"] > 0
        assert stats["relationships"] > 0

    # The model-invented entity types must have been dropped, not stored.
    stored_names = {entity["name"].lower() for entity in pipeline.graph_store.entities.values()}
    assert "mathematician" not in stored_names
    assert "energy" not in stored_names
    # A duplicate name with an invalid type does not erase the valid gold node.
    ohio = [
        entity
        for entity in pipeline.graph_store.entities.values()
        if entity["name"].lower() == "ohio"
    ]
    assert ohio and all(entity["type"] == "Location" for entity in ohio)

    for entity in pipeline.graph_store.entities.values():
        assert entity["type"] in APPROVED_ENTITY_TYPES

    for relationship in pipeline.graph_store.relationships:
        assert relationship["predicate"] in APPROVED_PREDICATES
        # Provenance points at a real chunk from the same page.
        matching = [
            chunk
            for chunk in pipeline.chunks[relationship["document_id"]]
            if chunk["chunk_id"] == relationship["chunk_id"]
        ]
        assert matching
        assert matching[0]["page_number"] == relationship["page_number"]
        assert relationship["evidence"]

    # The invalid predicate is preserved as RELATED_TO, never as INVENTED.
    predicates = {relationship["predicate"] for relationship in pipeline.graph_store.relationships}
    assert "INVENTED" not in predicates
    assert any(
        relationship["predicate"] == "RELATED_TO"
        and relationship["subject"] == "Charles Babbage"
        and relationship["object"] == "Analytical Engine"
        for relationship in pipeline.graph_store.relationships
    )


@pytest.mark.asyncio
async def test_graph_accuracy_against_gold_triples(tmp_path: Path):
    pipeline = await _pipeline(tmp_path)
    score = pipeline.graph_accuracy()

    assert score.recall == 1.0, f"missing gold triples: {score.missing}"
    assert score.precision >= 0.85, f"unexpected extra triples: {score.extra}"
    assert score.f1 >= 0.9
    # The only predicted-but-not-gold triples are the deliberate noise ones.
    assert len(score.extra) == 3


@pytest.mark.asyncio
async def test_graph_read_shape_matches_frontend_expectations(tmp_path: Path):
    pipeline = await _pipeline(tmp_path)
    graph = pipeline.graph_store.get_graph()

    assert graph["nodes"] and graph["edges"]
    labels = {node["label"] for node in graph["nodes"]}
    assert {"Ada Lovelace", "Charles Babbage", "NASA", "Photosynthesis"} <= labels
    assert all({"id", "label", "type"} <= set(node) for node in graph["nodes"])
    assert all(
        {"id", "source", "target", "label"} <= set(edge) for edge in graph["edges"]
    )
    assert all(edge["label"] in APPROVED_PREDICATES for edge in graph["edges"])


@pytest.mark.asyncio
async def test_deletion_removes_only_the_target_document(tmp_path: Path):
    pipeline = await _pipeline(tmp_path)
    apollo_before = pipeline.graph_store.counts("apollo11")
    assert apollo_before["relationships"] > 0

    cleanup = pipeline.delete_document("ada_lovelace")

    assert cleanup["vector_rows_removed"] > 0
    assert cleanup["chunks_removed"] > 0
    assert cleanup["file_removed"] is True
    assert cleanup["vector_rows_left"] == 0
    assert cleanup["chunks_left"] == 0
    assert cleanup["graph_left"] == {
        "entities": 0,
        "relationships": 0,
        "provenance": 0,
    }
    # Unrelated documents are untouched.
    assert pipeline.graph_store.counts("apollo11") == apollo_before
    assert pipeline.vector_store.count("apollo11") > 0

    for key in ("apollo11", "photosynthesis"):
        pipeline.delete_document(key)

    assert pipeline.vector_store.count() == 0
    assert pipeline.graph_store.counts() == {
        "entities": 0,
        "relationships": 0,
        "provenance": 0,
    }
    assert pipeline.chunks == {}
    assert pipeline.files == {}
