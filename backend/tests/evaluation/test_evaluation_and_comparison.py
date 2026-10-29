"""Evaluation metrics, vector-only vs GraphRAG comparison, latency, memory."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.evaluation.corpus import CORPUS, SourceRef
from tests.evaluation.metrics import (
    MemoryTracker,
    format_latency,
    is_abstention,
    score_citations,
    score_faithfulness,
    score_graph,
    score_retrieval,
)
from tests.evaluation.pipeline import DeterministicPipeline

pytestmark = pytest.mark.evaluation


@pytest.mark.asyncio
async def test_graphrag_retrieves_connected_fact_that_vector_window_misses(
    tmp_path: Path,
):
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()

    vector = await pipeline.evaluate_question("q_bridge", mode="vector", top_k=2)
    graphrag = await pipeline.evaluate_question("q_bridge", mode="graphrag", top_k=2)

    assert vector.retrieval_score is not None
    assert graphrag.retrieval_score is not None

    # Vector-only with a tight window never sees the answer page.
    assert vector.retrieval_score.recall == 0.0
    assert "trumpet" not in vector.answer.lower()
    assert is_abstention(vector.answer)
    assert vector.citations.total == 0

    # Graph evidence reaches the answer page through the Armstrong entity.
    # GraphRAG recall is measured over all returned evidence (the graph channel
    # is not limited by the vector window).
    assert graphrag.retrieval_score.recall == 1.0
    assert "trumpet" in graphrag.answer.lower()
    assert graphrag.citations.precision == 1.0
    assert graphrag.faithfulness.score == 1.0

    # The graph path also returns more evidence (higher recall, lower precision
    # is an accepted trade-off; see docs/EVALUATION.md).
    assert len(graphrag.evidence) > len(vector.evidence)


@pytest.mark.asyncio
async def test_vector_only_is_sufficient_for_single_hop_questions(tmp_path: Path):
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()

    for key in ("q_babbage", "q_apollo", "q_multi"):
        vector = await pipeline.evaluate_question(key, mode="vector", top_k=2)
        graphrag = await pipeline.evaluate_question(key, mode="graphrag", top_k=2)
        assert vector.retrieval_score is not None
        assert graphrag.retrieval_score is not None
        assert vector.retrieval_score.recall == graphrag.retrieval_score.recall == 1.0
        assert vector.answer_keyword_hit
        assert graphrag.answer_keyword_hit


@pytest.mark.asyncio
async def test_comparison_report_is_deterministic(tmp_path: Path):
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()

    rows = []
    for question in CORPUS.questions:
        if not question.expected_sources:
            continue
        vector = await pipeline.evaluate_question(question.key, mode="vector", top_k=2)
        graphrag = await pipeline.evaluate_question(question.key, mode="graphrag", top_k=2)
        rows.append(
            (
                question.key,
                vector.retrieval_score.recall,
                graphrag.retrieval_score.recall,
                vector.answer_keyword_hit,
                graphrag.answer_keyword_hit,
            )
        )

    # Every connected-fact question must be answered correctly by GraphRAG.
    connected = {
        question.key for question in CORPUS.questions if question.requires_graph
    }
    for key, _vector_recall, graph_recall, _vector_ok, graph_ok in rows:
        if key in connected:
            assert graph_recall == 1.0, key
            assert graph_ok, key

    # GraphRAG must never retrieve a subset of the vector evidence.
    for question in CORPUS.questions:
        if not question.expected_sources:
            continue
        vector = await pipeline.evaluate_question(question.key, mode="vector", top_k=2)
        graphrag = await pipeline.evaluate_question(question.key, mode="graphrag", top_k=2)
        assert graphrag.retrieval_score.recall >= vector.retrieval_score.recall

    print("\nvector-only vs GraphRAG (vector window top_k=2; GraphRAG recall over all evidence)")
    print(f"{'question':<12} {'vec recall':>10} {'graph recall':>12} {'vec ok':>7} {'graph ok':>8}")
    for key, vector_recall, graph_recall, vector_ok, graph_ok in rows:
        print(
            f"{key:<12} {vector_recall:>10.2f} {graph_recall:>12.2f} "
            f"{str(vector_ok):>7} {str(graph_ok):>8}"
        )


@pytest.mark.asyncio
async def test_metrics_reject_fabricated_citations_and_claims(tmp_path: Path):
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()

    # The generator drops citation IDs that are not in the evidence, so the
    # answer is left with a claim and zero valid citations.
    invalid = await pipeline.evaluate_question("q_ada", llm_mode="invalid_citation")
    assert invalid.citations.total == 0
    assert invalid.citations.precision == 0.0
    assert invalid.faithfulness.score == 0.0
    # The claim itself is present, but it has no valid citation.
    assert "ada lovelace" in invalid.answer.lower()

    # A direct metric call also flags the unknown ID.
    direct = score_citations(
        [
            {
                "chunk_id": "missing-chunk-id",
                "document_id": "ada_lovelace",
                "page_number": 2,
                "snippet": "",
            }
        ],
        invalid.evidence,
    )
    assert direct.invalid_ids == ("missing-chunk-id",)
    assert direct.precision == 0.0

    # A real chunk ID does not make an unsupported sentence faithful.
    unsupported = await pipeline.evaluate_question("q_ada", llm_mode="unsupported_claim")
    assert unsupported.citations.precision == 1.0
    assert unsupported.faithfulness.score == 0.0
    assert not unsupported.answer_keyword_hit


def test_retrieval_metric_ranks_expected_sources():
    expected = [SourceRef("doc_a", 1), SourceRef("doc_b", 2)]
    retrieved = [
        SourceRef("doc_c", 1),
        SourceRef("doc_b", 2),
        SourceRef("doc_a", 1),
    ]
    score = score_retrieval(retrieved, expected, k=3)
    assert score is not None
    assert score.hit_at_k
    assert score.recall == 1.0
    assert score.reciprocal_rank == pytest.approx(0.5)
    assert score.precision == pytest.approx(2 / 3)

    limited = score_retrieval(retrieved, expected, k=1)
    assert limited is not None and not limited.hit_at_k
    assert score_retrieval(retrieved, [], k=3) is None


def test_faithfulness_ignores_uncited_abstention():
    report = score_faithfulness(
        "Insufficient evidence to answer this question.",
        [],
        [],
    )
    assert report.abstained
    assert report.score == 1.0


def test_graph_metric_reports_missing_and_extra_triples():
    gold = [("Ada Lovelace", "RELATED_TO", "Charles Babbage")]
    predicted = [
        ("ada lovelace", "related_to", "charles babbage"),
        ("Ada Lovelace", "CREATED", "Analytical Engine"),
    ]
    score = score_graph(predicted, gold)
    assert score.recall == 1.0
    assert score.precision == 0.5
    assert ("ada lovelace", "CREATED", "analytical engine") in score.extra
    assert score_graph([], gold).missing == (("ada lovelace", "RELATED_TO", "charles babbage"),)


@pytest.mark.asyncio
async def test_latency_and_memory_are_recorded(tmp_path: Path):
    with MemoryTracker() as memory:
        pipeline = DeterministicPipeline(CORPUS, tmp_path)
        await pipeline.ingest_all()
        runs = await pipeline.evaluate_all()

    assert len(runs) == len(CORPUS.questions)
    summary = pipeline.latency.summary()
    assert {"validate_pdf", "extract_pages", "chunk", "index_embeddings", "extract_graph"} <= set(
        summary
    )
    assert summary["retrieve"]["count"] == float(len(CORPUS.questions))
    assert memory.peak_mb < 256, f"unexpected peak allocation: {memory.peak_mb:.1f} MB"

    report = format_latency(summary)
    print(f"\nlatency (deterministic doubles, {len(CORPUS.documents)} docs)")
    print(report)
    print(f"peak traced memory: {memory.peak_mb:.2f} MB")
