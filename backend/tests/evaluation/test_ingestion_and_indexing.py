"""Synthetic ingestion: PDF -> pages -> chunks -> embeddings (steps 1-4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import settings
from tests.evaluation.corpus import CORPUS, SourceRef
from tests.evaluation.doubles import cosine
from tests.evaluation.metrics import score_retrieval
from tests.evaluation.pipeline import DeterministicPipeline

pytestmark = pytest.mark.evaluation


@pytest.mark.asyncio
async def test_synthetic_pdfs_produce_page_level_chunks_and_embeddings(tmp_path: Path):
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    stats = await pipeline.ingest_all()

    assert set(stats) == {"ada_lovelace", "apollo11", "photosynthesis"}
    total_pages = sum(len(document.pages) for document in CORPUS.documents)
    total_chunks = sum(item["pages"] for item in stats.values())
    assert total_chunks == total_pages

    for document in CORPUS.documents:
        assert pipeline.files[document.key].read_bytes().startswith(b"%PDF")
        chunks = pipeline.chunks[document.key]
        assert {chunk["page_number"] for chunk in chunks} == {
            page.page_number for page in document.pages
        }
        assert all(chunk["chunk_text"].strip() for chunk in chunks)
        assert all(chunk["char_count"] > 0 for chunk in chunks)

        stored = [
            row
            for row in pipeline.vector_store.rows
            if row["document_id"] == document.key
        ]
        assert len(stored) == len(chunks)
        for row in stored:
            assert len(row["embedding"]) == settings.EMBEDDING_DIMENSION
            assert abs(cosine(row["embedding"], row["embedding"]) - 1.0) < 1e-9


@pytest.mark.asyncio
async def test_embeddings_are_deterministic_and_searchable(tmp_path: Path):
    first = DeterministicPipeline(CORPUS, tmp_path / "first")
    await first.ingest_all()

    second = DeterministicPipeline(CORPUS, tmp_path / "second")
    await second.ingest_all()

    first_vector = first.vector_store.rows[0]["embedding"]
    second_vector = second.vector_store.rows[0]["embedding"]
    assert first_vector == second_vector

    results = await first.vector_store.search_similar(
        "Who designed the Analytical Engine?", top_k=5
    )
    score = score_retrieval(
        [SourceRef(row["document_id"], row["page_number"]) for row in results],
        [SourceRef("ada_lovelace", 1)],
        k=5,
    )
    assert score is not None and score.hit_at_k
    assert results[0]["document_id"] == "ada_lovelace"
    assert results[0]["page_number"] == 1


@pytest.mark.asyncio
async def test_vector_search_is_document_scoped(tmp_path: Path):
    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()

    scoped = await pipeline.vector_store.search_similar(
        "Who designed the Analytical Engine?",
        top_k=10,
        document_id="photosynthesis",
    )
    assert scoped
    assert all(row["document_id"] == "photosynthesis" for row in scoped)
    assert all("babbage" not in row["text"].lower() for row in scoped)


@pytest.mark.asyncio
async def test_bridge_question_is_missed_by_tight_vector_window(tmp_path: Path):
    """Precondition for the vector-only vs GraphRAG comparison.

    The answer page (Apollo page 3) intentionally shares almost no vocabulary
    with the question, so a two-chunk vector window cannot reach it.
    """

    pipeline = DeterministicPipeline(CORPUS, tmp_path)
    await pipeline.ingest_all()

    results = await pipeline.vector_store.search_similar(
        "Which instrument did the first person to walk on the Moon play?",
        top_k=2,
    )
    sources = [SourceRef(row["document_id"], row["page_number"]) for row in results]
    assert SourceRef("apollo11", 3) not in sources
    assert all("trumpet" not in row["text"].lower() for row in results)
