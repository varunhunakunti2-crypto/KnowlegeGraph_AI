"""End-to-end GraphRAG flow on the synthetic corpus (steps 7-12)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import settings
from tests.evaluation.corpus import CORPUS
from tests.evaluation.metrics import is_abstention
from tests.evaluation.pipeline import DeterministicPipeline

pytestmark = pytest.mark.evaluation

ANSWERABLE = ("q_babbage", "q_ada", "q_1843", "q_apollo", "q_bridge", "q_multi")


async def _run_all(tmp_path: Path) -> dict:
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()
    runs = await pipeline.evaluate_all(top_k=settings.TOP_K_CHUNKS)
    return {run.question.key: run for run in runs}


@pytest.mark.asyncio
async def test_every_answerable_question_gets_a_cited_grounded_answer(tmp_path: Path):
    runs = await _run_all(tmp_path)

    for key in ANSWERABLE:
        run = runs[key]
        assert run.answer_keyword_hit, f"{key}: {run.answer!r}"
        assert run.citations.total >= 1, key
        assert run.citations.precision == 1.0, (
            f"{key}: invalid citations {run.citations.invalid_ids}"
        )
        assert not run.citations.invalid_ids
        assert run.faithfulness.score >= 0.9, (
            f"{key}: unsupported sentences {run.faithfulness.unsupported}"
        )
        assert not run.faithfulness.abstained
        assert run.retrieval_score is not None
        assert run.retrieval_score.recall == 1.0, (
            f"{key}: expected {run.retrieval_score.expected} "
            f"got {run.retrieval_score.retrieved}"
        )
        assert not run.forbidden_keyword_hit


@pytest.mark.asyncio
async def test_evidence_combines_vector_and_graph_sources(tmp_path: Path):
    runs = await _run_all(tmp_path)
    run = runs["q_ada"]

    source_types = {item["source_type"] for item in run.evidence}
    assert {"vector", "graph"} <= source_types
    # Graph evidence carries relationship provenance, not just free text.
    assert any(item.get("entity_id") for item in run.evidence)
    assert any(item["source_type"] == "graph" for item in run.evidence)


@pytest.mark.asyncio
async def test_missing_evidence_abstains_without_fabricating_citations(tmp_path: Path):
    runs = await _run_all(tmp_path)
    run = runs["q_missing"]

    # Known limitation: retrieval has no relevance floor, so unrelated chunks
    # are still returned. The generator must refuse to answer anyway.
    assert run.retrieval["has_evidence"] is True
    assert is_abstention(run.answer)
    assert run.citations.total == 0
    assert run.faithfulness.abstained
    assert "france" not in run.answer.lower()


@pytest.mark.asyncio
async def test_document_scope_prevents_cross_document_leakage(tmp_path: Path):
    runs = await _run_all(tmp_path)
    run = runs["q_scope"]

    assert is_abstention(run.answer), run.answer
    assert not run.forbidden_keyword_hit
    assert run.citations.total == 0
    assert all(
        item["document_id"] == "photosynthesis" for item in run.evidence
    ), "scoped retrieval leaked evidence from another document"


@pytest.mark.asyncio
async def test_graph_accuracy_and_deletion_cleanup(tmp_path: Path):
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()

    graph_score = pipeline.graph_accuracy()
    assert graph_score.f1 >= 0.9
    assert graph_score.recall == 1.0

    cleanup = pipeline.delete_document("apollo11")
    assert cleanup["vector_rows_left"] == 0
    assert cleanup["graph_left"] == {"entities": 0, "relationships": 0, "provenance": 0}
    assert cleanup["file_removed"]
    assert pipeline.files.get("apollo11") is None

    for key in ("ada_lovelace", "photosynthesis"):
        pipeline.delete_document(key)

    assert pipeline.vector_store.count() == 0
    assert pipeline.graph_store.counts() == {
        "entities": 0,
        "relationships": 0,
        "provenance": 0,
    }
    assert all(not path.exists() for path in pipeline.files.values())

    # After deletion, the same question must report no evidence at all.
    result = await pipeline.evaluate_question("q_babbage")
    assert result.retrieval["has_evidence"] is False
    assert is_abstention(result.answer)
    assert result.citations.total == 0
    assert result.faithfulness.abstained
