"""In-process end-to-end GraphRAG pipeline for deterministic evaluation.

Runs the real application components:

    synthetic PDF -> PDFValidator -> PDFExtractor -> TextChunker
        -> deterministic embeddings -> in-memory vector store
        -> GraphExtractionService (scripted structured JSON) -> in-memory graph
        -> GraphRAGRetriever (vector + graph evidence) -> AnswerGenerator
        -> citation validation -> cleanup

Only the external boundaries are replaced: the embedding model, the LLM, the
SQL database, and Neo4j. Everything between them is production code.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings
from app.generation.answer import AnswerGenerator
from app.ingestion.chunker import TextChunker
from app.ingestion.extractor import PDFExtractor
from app.ingestion.validator import PDFValidator
from app.retrieval.graphrag import GraphRAGRetriever
from app.services.graph_extraction import GraphExtractionService

from tests.evaluation.corpus import (
    EvalQuestion,
    GoldEntity,
    GoldRelationship,
    SampleDocument,
    SamplePage,
    SourceRef,
    SyntheticCorpus,
)
from tests.evaluation.doubles import (
    DeterministicEmbedder,
    FixedResponseLLM,
    InMemoryGraphStore,
    InMemoryVectorStore,
    ScriptedAnswerLLM,
)
from tests.evaluation.metrics import (
    CitationReport,
    FaithfulnessReport,
    LatencyRecorder,
    RetrievalScore,
    score_citations,
    score_faithfulness,
    score_graph,
    score_retrieval,
)
from tests.evaluation.pdfs import write_pdf

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def _entity_evidence(page: SamplePage, entity: GoldEntity) -> str:
    needle = entity.name.lower()
    for sentence in _sentences(page.text):
        if needle in sentence.lower():
            return sentence
    return page.text[:500]


def _extraction_payload(page: SamplePage) -> Dict[str, Any]:
    """Build the structured JSON that the scripted extractor returns.

    Gold annotations are returned plus deliberate noise, so the sanitiser and
    the graph-accuracy metric both have real work to do.
    """

    entities = [
        {
            "name": entity.name,
            "type": entity.type,
            "aliases": [],
            "confidence": 0.9,
            "evidence": _entity_evidence(page, entity),
        }
        for entity in list(page.entities) + list(page.noise_entities)
    ]
    relationships = [
        {
            "subject": relation.subject,
            "predicate": relation.predicate,
            "object": relation.object,
            "confidence": 0.85,
            "evidence": relation.evidence,
        }
        for relation in list(page.relationships) + list(page.noise_relationships)
    ]
    return {"entities": entities, "relationships": relationships}


@dataclass
class QuestionRun:
    """One retrieval + generation run for one question."""

    question: EvalQuestion
    mode: str
    retrieval: Dict[str, Any]
    generation: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    k: int
    window: int
    retrieval_score: Optional[RetrievalScore]
    citations: CitationReport
    faithfulness: FaithfulnessReport
    answer_keyword_hit: bool
    forbidden_keyword_hit: bool

    @property
    def answer(self) -> str:
        return str(self.generation.get("answer") or "")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "question_key": self.question.key,
            "question": self.question.question,
            "mode": self.mode,
            "document_scope": self.question.document_scope,
            "retrieved_sources": [
                [item["document_id"], item["page_number"]] for item in self.evidence
            ],
            "has_evidence": self.retrieval["has_evidence"],
            "answer": self.answer,
            "answer_keyword_hit": self.answer_keyword_hit,
            "forbidden_keyword_hit": self.forbidden_keyword_hit,
            "retrieval": None
            if self.retrieval_score is None
            else {
                "hit_at_k": self.retrieval_score.hit_at_k,
                "recall": self.retrieval_score.recall,
                "precision": self.retrieval_score.precision,
                "reciprocal_rank": self.retrieval_score.reciprocal_rank,
                "k": self.k,
                "window": self.window,
            },
            "citations": {
                "total": self.citations.total,
                "valid": self.citations.valid,
                "precision": self.citations.precision,
                "invalid_ids": list(self.citations.invalid_ids),
            },
            "faithfulness": {
                "score": self.faithfulness.score,
                "sentences": self.faithfulness.sentences,
                "supported": self.faithfulness.supported,
                "unsupported": list(self.faithfulness.unsupported),
                "abstained": self.faithfulness.abstained,
            },
        }


class DeterministicPipeline:
    """Evaluate the GraphRAG pipeline without external services."""

    def __init__(
        self,
        corpus: SyntheticCorpus,
        workdir: Path,
        default_top_k: Optional[int] = None,
    ):
        self.corpus = corpus
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.default_top_k = default_top_k or settings.TOP_K_CHUNKS

        texts = [
            page.text for document in corpus.documents for page in document.pages
        ]
        self.embedder = DeterministicEmbedder(texts, dimension=settings.EMBEDDING_DIMENSION)
        self.vector_store = InMemoryVectorStore(self.embedder)
        self.graph_store = InMemoryGraphStore()
        self.latency = LatencyRecorder()

        self.chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.files: Dict[str, Path] = {}
        self.ingest_stats: Dict[str, Dict[str, int]] = {}
        # Only questions with an expected answer phrase can drive the extractive
        # double; the same question text may appear twice with different scopes.
        self.answer_keywords = {
            question.question: tuple(question.answer_keywords)
            for question in corpus.questions
            if question.answer_keywords
        }

        self._retriever: Optional[GraphRAGRetriever] = None

    # -- ingestion ---------------------------------------------------------

    async def ingest(self, document: SampleDocument) -> Dict[str, int]:
        """Run the real upload-equivalent ingestion for one synthetic PDF."""

        pdf_path = write_pdf(self.workdir / document.filename, document)
        self.files[document.key] = pdf_path

        with self.latency.time("validate_pdf"):
            validation = PDFValidator.validate_upload(
                str(pdf_path),
                "application/pdf",
                settings.MAX_UPLOAD_SIZE_MB,
                settings.MAX_PDF_PAGES,
                settings.ALLOW_PDF_ACTIVE_CONTENT,
            )
        assert validation["page_count"] == len(document.pages)

        with self.latency.time("extract_pages"):
            pages = PDFExtractor.extract_text(
                str(pdf_path),
                settings.MAX_PDF_PAGES,
                settings.MAX_EXTRACTED_TEXT_CHARS,
            )

        with self.latency.time("chunk"):
            chunker = TextChunker(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            chunks = chunker.chunk_pages(pages)

        indexed = 0
        extracted_entities = 0
        extracted_relationships = 0

        for chunk in chunks:
            chunk_id = f"chunk-{document.key}-{chunk['page_number']}-{chunk['chunk_index']}"
            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": document.key,
                "page_number": int(chunk["page_number"]),
                "chunk_index": int(chunk["chunk_index"]),
                "chunk_text": chunk["chunk_text"],
                "char_count": int(chunk["char_count"]),
            }
            self.chunks[document.key].append(chunk_record)

            with self.latency.time("index_embeddings"):
                embedding = self.embedder.embed(chunk_record["chunk_text"])
                self.vector_store.index(
                    chunk_id=chunk_id,
                    document_id=document.key,
                    page_number=chunk_record["page_number"],
                    chunk_index=chunk_record["chunk_index"],
                    text=chunk_record["chunk_text"],
                    embedding=embedding,
                )
                indexed += 1

            page = document.page(chunk_record["page_number"])
            payload = json.dumps(_extraction_payload(page))
            service = GraphExtractionService()
            service.llm = FixedResponseLLM(payload)

            with self.latency.time("extract_graph"):
                extraction = await service.extract_from_chunk(chunk_record)
                stored = self.graph_store.store_extraction(document.key, extraction)
            extracted_entities += stored["entities_stored"]
            extracted_relationships += stored["relationships_stored"]

        self.ingest_stats[document.key] = {
            "pages": len(pages),
            "chunks": len(chunks),
            "indexed": indexed,
            "entities": extracted_entities,
            "relationships": extracted_relationships,
        }
        return self.ingest_stats[document.key]

    async def ingest_all(self) -> Dict[str, Dict[str, int]]:
        for document in self.corpus.documents:
            await self.ingest(document)
        return self.ingest_stats

    # -- retrieval ---------------------------------------------------------

    def _graph_lookup(
        self, question: str, document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Question-independent graph evidence, matching production behaviour."""

        return self.graph_store.retrieve(document_id)

    @property
    def retriever(self) -> GraphRAGRetriever:
        if self._retriever is None:
            retriever = GraphRAGRetriever()
            retriever.vector_db = self.vector_store  # type: ignore[assignment]
            retriever._graph_retrieval = self._graph_lookup  # type: ignore[method-assign]
            self._retriever = retriever
        return self._retriever

    async def retrieve(
        self,
        question: str,
        *,
        mode: str = "graphrag",
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        limit = top_k or self.default_top_k
        if mode == "graphrag":
            with self.latency.time("retrieve"):
                return await self.retriever.retrieve(
                    question, document_id=document_id, top_k=limit
                )
        if mode == "vector":
            with self.latency.time("retrieve"):
                results = await self.vector_store.search_similar(
                    question, top_k=limit, document_id=document_id
                )
            evidence = [
                {
                    "source_type": "vector",
                    "chunk_id": result["chunk_id"],
                    "document_id": result["document_id"],
                    "page_number": result["page_number"],
                    "text": result["text"],
                    "score": result["similarity"] * 0.6,
                    "entity_id": None,
                    "relationship_info": None,
                }
                for result in results
            ]
            return {
                "evidence": evidence,
                "has_evidence": bool(evidence),
                "context": "\n\n".join(item["text"] for item in evidence),
                "total_tokens_estimate": sum(len(item["text"]) for item in evidence) // 4,
            }
        raise ValueError(f"Unknown retrieval mode: {mode}")

    # -- generation --------------------------------------------------------

    async def answer(
        self,
        question: str,
        evidence: Sequence[Dict[str, Any]],
        *,
        mode: str = "correct",
    ) -> Dict[str, Any]:
        generator = AnswerGenerator()
        generator.llm = ScriptedAnswerLLM(self.answer_keywords, mode=mode)  # type: ignore[assignment]
        with self.latency.time("generate"):
            return await generator.generate_answer(question, list(evidence))

    # -- evaluation --------------------------------------------------------

    def _source_refs(self, evidence: Sequence[Dict[str, Any]]) -> List[SourceRef]:
        return [
            SourceRef(str(item["document_id"]), int(item.get("page_number") or 0))
            for item in evidence
        ]

    async def evaluate_question(
        self,
        question: EvalQuestion | str,
        *,
        mode: str = "graphrag",
        top_k: Optional[int] = None,
        llm_mode: str = "correct",
    ) -> QuestionRun:
        if isinstance(question, str):
            question = self.corpus.question(question)

        limit = top_k or self.default_top_k
        retrieval = await self.retrieve(
            question.question,
            mode=mode,
            document_id=question.document_scope,
            top_k=limit,
        )
        generation = await self.answer(
            question.question,
            retrieval["evidence"],
            mode=llm_mode,
        )
        evidence = [dict(item) for item in retrieval["evidence"]]

        # Vector-only retrieval is bounded by the window, but the graph channel
        # is not: it appends relationship evidence in addition to the vector
        # results. Score recall over everything the generator actually sees, and
        # keep ``window`` for the vector top-k comparison.
        evidence_k = max(1, len(evidence))
        retrieval_score = score_retrieval(
            self._source_refs(evidence),
            question.expected_sources,
            evidence_k,
        )
        citation_report = score_citations(
            generation.get("citations", []),
            evidence,
            question.expected_sources,
        )
        faithfulness = score_faithfulness(
            generation.get("answer", ""),
            generation.get("citations", []),
            evidence,
        )
        answer = str(generation.get("answer") or "").lower()
        keyword_hit = all(
            keyword.lower() in answer for keyword in question.answer_keywords
        )
        forbidden_hit = any(
            keyword.lower() in answer for keyword in question.forbidden_keywords
        )

        return QuestionRun(
            question=question,
            mode=mode,
            retrieval=retrieval,
            generation=generation,
            evidence=evidence,
            k=evidence_k,
            window=limit,
            retrieval_score=retrieval_score,
            citations=citation_report,
            faithfulness=faithfulness,
            answer_keyword_hit=keyword_hit,
            forbidden_keyword_hit=forbidden_hit,
        )

    async def evaluate_all(
        self,
        *,
        mode: str = "graphrag",
        top_k: Optional[int] = None,
    ) -> List[QuestionRun]:
        runs = []
        for question in self.corpus.questions:
            runs.append(await self.evaluate_question(question, mode=mode, top_k=top_k))
        return runs

    def graph_accuracy(self):
        """Score stored triples against the hand-annotated gold triples."""

        return score_graph(
            self.graph_store.relation_triples(),
            [
                (relation.subject, relation.predicate, relation.object)
                for relation in self.corpus.gold_relationships()
            ],
        )

    # -- cleanup -----------------------------------------------------------

    def delete_document(self, document_key: str) -> Dict[str, Any]:
        """Delete one document exactly as the API promises, and report leftovers."""

        vector_removed = self.vector_store.delete_document(document_key)
        self.graph_store.delete_document_graph(document_key)
        chunks_removed = len(self.chunks.pop(document_key, []))

        file_path = self.files.pop(document_key, None)
        file_removed = False
        if file_path is not None and file_path.exists():
            file_path.unlink()
            file_removed = True

        return {
            "vector_rows_removed": vector_removed,
            "chunks_removed": chunks_removed,
            "file_removed": file_removed,
            "vector_rows_left": self.vector_store.count(document_key),
            "chunks_left": len(self.chunks.get(document_key, [])),
            "graph_left": self.graph_store.counts(document_key),
        }
