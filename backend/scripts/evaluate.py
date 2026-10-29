#!/usr/bin/env python
r"""Deterministic GraphRAG evaluation report (no models, no databases).

Runs the production ingestion, graph extraction, retrieval, generation,
citation, and deletion code against the synthetic corpus in
``tests/evaluation``. Only the external boundaries are replaced:

    embeddings   -> TF-IDF hashed embedder
    vector store -> in-memory list
    graph store  -> in-memory store with the same provenance rules
    LLM          -> scripted extractor / extractive answerer

Nothing here measures qwen3:4b or nomic-embed-text quality. Use the optional
live suite (``KG_LIVE=1``) for that.

Usage (from backend/):

    .\.venv\Scripts\python.exe -m scripts.evaluate
    .\.venv\Scripts\python.exe -m scripts.evaluate --top-k 2 --json eval-report.json

Exit code is 0 only when every check passes (``--no-strict`` reports failures
but still exits 0).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings

from tests.evaluation.corpus import CORPUS
from tests.evaluation.metrics import (
    MemoryTracker,
    format_latency,
    is_abstention,
)
from tests.evaluation.pipeline import DeterministicPipeline, QuestionRun


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _source_list(run: QuestionRun) -> str:
    return ", ".join(
        f"{item['document_id']}:p{item['page_number']}" for item in run.evidence
    ) or "-"


def _one_line(text: str, limit: int = 110) -> str:
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "..."


def _question_row(run: QuestionRun) -> Dict[str, Any]:
    payload = run.as_dict()
    payload["evidence_count"] = len(run.evidence)
    abstained = is_abstention(run.answer)
    payload["abstained"] = abstained
    # Questions with an answer phrase must answer; questions without one
    # (out-of-corpus / scoped-abstention controls) must abstain.
    if run.question.answer_keywords:
        payload["answer_ok"] = run.answer_keyword_hit and not abstained
    else:
        payload["answer_ok"] = abstained and not run.forbidden_keyword_hit
    payload["citation_precision"] = run.citations.precision
    payload["citation_expected_recall"] = run.citations.expected_source_recall
    payload["faithfulness_score"] = run.faithfulness.score
    return payload


def _comparison_row(
    question_key: str,
    vector: QuestionRun,
    graphrag: QuestionRun,
) -> Dict[str, Any]:
    def recall(run: QuestionRun):
        return None if run.retrieval_score is None else run.retrieval_score.recall

    def precision(run: QuestionRun):
        return None if run.retrieval_score is None else run.retrieval_score.precision

    return {
        "question_key": question_key,
        "vector": {
            "recall": recall(vector),
            "precision": precision(vector),
            "answer_keyword_hit": vector.answer_keyword_hit,
            "abstained": is_abstention(vector.answer),
            "evidence_count": len(vector.evidence),
            "citations": vector.citations.total,
        },
        "graphrag": {
            "recall": recall(graphrag),
            "precision": precision(graphrag),
            "answer_keyword_hit": graphrag.answer_keyword_hit,
            "abstained": is_abstention(graphrag.answer),
            "evidence_count": len(graphrag.evidence),
            "citations": graphrag.citations.total,
        },
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


async def build_report(top_k: int) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="kg-eval-") as temp_dir:
        with MemoryTracker() as memory:
            pipeline = DeterministicPipeline(CORPUS, Path(temp_dir), default_top_k=top_k)
            ingest_stats = await pipeline.ingest_all()

            vector_runs = [
                await pipeline.evaluate_question(question.key, mode="vector", top_k=top_k)
                for question in CORPUS.questions
            ]
            graphrag_runs = [
                await pipeline.evaluate_question(
                    question.key, mode="graphrag", top_k=top_k
                )
                for question in CORPUS.questions
            ]

        graph_score = pipeline.graph_accuracy()
        latency = pipeline.latency.summary()
        cleanup = pipeline.delete_document(CORPUS.documents[-1].key)

        by_key = {run.question.key: run for run in graphrag_runs}
        vector_by_key = {run.question.key: run for run in vector_runs}

        # -- checks --------------------------------------------------------
        check(
            "ingestion indexes every extracted chunk",
            all(
                stats["indexed"] == stats["chunks"] > 0
                for stats in ingest_stats.values()
            ),
            str({key: stats["chunks"] for key, stats in ingest_stats.items()}),
        )
        check(
            "graph extraction recovers every gold relationship",
            math.isclose(graph_score.recall, 1.0),
            f"recall={graph_score.recall:.2f} predicted={graph_score.predicted_count} "
            f"gold={graph_score.gold_count}",
        )
        check(
            "all citations point at retrieved evidence",
            all(
                not run.citations.invalid_ids
                for run in graphrag_runs
            ),
            f"invalid={[list(run.citations.invalid_ids) for run in graphrag_runs if run.citations.invalid_ids]}",
        )
        connected = [
            question for question in CORPUS.questions if question.requires_graph
        ]
        check(
            "GraphRAG answers connected-fact questions",
            all(
                by_key[question.key].retrieval_score is not None
                and by_key[question.key].retrieval_score.recall == 1.0
                and by_key[question.key].answer_keyword_hit
                for question in connected
            ),
            ", ".join(question.key for question in connected),
        )
        single_hop = [
            question
            for question in CORPUS.questions
            if question.expected_sources and not question.requires_graph
        ]
        check(
            "GraphRAG answers single-hop questions",
            all(by_key[question.key].answer_keyword_hit for question in single_hop),
            ", ".join(question.key for question in single_hop),
        )
        check(
            "out-of-corpus question abstains without citations",
            is_abstention(by_key["q_missing"].answer)
            and by_key["q_missing"].citations.total == 0,
            _one_line(by_key["q_missing"].answer),
        )
        check(
            "document scope does not leak entities",
            is_abstention(by_key["q_scope"].answer)
            and not by_key["q_scope"].forbidden_keyword_hit,
            _one_line(by_key["q_scope"].answer),
        )
        check(
            "faithfulness of every non-abstained answer",
            all(
                run.faithfulness.score == 1.0
                for run in graphrag_runs
                if not is_abstention(run.answer)
            ),
            ", ".join(
                f"{run.question.key}={run.faithfulness.score:.2f}"
                for run in graphrag_runs
                if not is_abstention(run.answer)
            ),
        )
        check(
            "deletion removes vector, graph, chunk, and file data",
            cleanup["vector_rows_left"] == 0
            and cleanup["chunks_left"] == 0
            and cleanup["graph_left"] == {"entities": 0, "relationships": 0, "provenance": 0}
            and cleanup["file_removed"],
            str(cleanup),
        )
        check(
            "vector-only misses the connected fact at the same window",
            vector_by_key["q_bridge"].retrieval_score is not None
            and vector_by_key["q_bridge"].retrieval_score.recall == 0.0
            and is_abstention(vector_by_key["q_bridge"].answer),
            _one_line(vector_by_key["q_bridge"].answer),
        )

    question_rows = [_question_row(run) for run in graphrag_runs]
    comparison_rows = [
        _comparison_row(question.key, vector_by_key[question.key], by_key[question.key])
        for question in CORPUS.questions
        if question.expected_sources
    ]

    return {
        "suite": "deterministic",
        "note": (
            "TF-IDF embedder, in-memory stores, scripted extraction/answer LLM. "
            "Production ingestion, retrieval, generation and deletion code is exercised."
        ),
        "top_k": top_k,
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "production_top_k_chunks": settings.TOP_K_CHUNKS,
        },
        "corpus": {
            "documents": [
                {
                    "key": document.key,
                    "filename": document.filename,
                    "pages": len(document.pages),
                }
                for document in CORPUS.documents
            ],
            "questions": [question.key for question in CORPUS.questions],
        },
        "ingestion": ingest_stats,
        "graph_accuracy": {
            "precision": graph_score.precision,
            "recall": graph_score.recall,
            "f1": graph_score.f1,
            "predicted_count": graph_score.predicted_count,
            "gold_count": graph_score.gold_count,
            "missing": [list(triple) for triple in graph_score.missing],
            "extra": [list(triple) for triple in graph_score.extra],
        },
        "graphrag_questions": question_rows,
        "comparison": comparison_rows,
        "latency": latency,
        "memory_peak_mb": memory.peak_mb,
        "cleanup": cleanup,
        "checks": checks,
    }


def print_report(report: Dict[str, Any]) -> None:
    print("KnowledgeGraph AI - deterministic GraphRAG evaluation")
    print(f"suite: {report['suite']} (top_k={report['top_k']})")
    print(report["note"])
    print()

    print("corpus")
    for document in report["corpus"]["documents"]:
        print(
            f"  {document['key']:<16} {document['pages']} pages  "
            f"{document['filename']}"
        )
    print()

    print("ingestion")
    print(
        f"  {'document':<16} {'pages':>5} {'chunks':>7} {'indexed':>8} "
        f"{'entities':>9} {'relations':>10}"
    )
    for key, stats in report["ingestion"].items():
        print(
            f"  {key:<16} {stats['pages']:>5} {stats['chunks']:>7} "
            f"{stats['indexed']:>8} {stats['entities']:>9} {stats['relationships']:>10}"
        )
    print()

    graph = report["graph_accuracy"]
    print("graph accuracy (gold triples, including deliberate model noise)")
    print(
        f"  precision={graph['precision']:.2f} recall={graph['recall']:.2f} "
        f"f1={graph['f1']:.2f} predicted={graph['predicted_count']} "
        f"gold={graph['gold_count']}"
    )
    for triple in graph["missing"]:
        print(f"  missing: {triple}")
    print()

    print("GraphRAG questions")
    print(
        f"  {'question':<12} {'evid':>4} {'recall':>6} {'prec':>6} "
        f"{'answer':>6} {'cites':>5} {'faith':>5}  answer"
    )
    for row in report["graphrag_questions"]:
        retrieval = row["retrieval"] or {}
        print(
            f"  {row['question_key']:<12} {row['evidence_count']:>4} "
            f"{_fmt(retrieval.get('recall')):>6} {_fmt(retrieval.get('precision')):>6} "
            f"{'yes' if row['answer_ok'] else 'no':>6} "
            f"{row['citations']['valid']}/{row['citations']['total']:<3} "
            f"{_fmt(row['faithfulness']['score']):>5}  {_one_line(row['answer'], 90)}"
        )
    print()

    print(f"vector-only vs GraphRAG (retrieval window top_k={report['top_k']})")
    print(
        f"  {'question':<12} {'vec rec':>7} {'gr rec':>7} {'vec prec':>8} "
        f"{'gr prec':>7} {'vec ok':>6} {'gr ok':>5} {'vec n':>5} {'gr n':>4}"
    )
    for row in report["comparison"]:
        vector = row["vector"]
        graphrag = row["graphrag"]
        print(
            f"  {row['question_key']:<12} "
            f"{_fmt(vector['recall']):>7} {_fmt(graphrag['recall']):>7} "
            f"{_fmt(vector['precision']):>8} {_fmt(graphrag['precision']):>7} "
            f"{'yes' if vector['answer_keyword_hit'] else 'no':>6} "
            f"{'yes' if graphrag['answer_keyword_hit'] else 'no':>5} "
            f"{vector['evidence_count']:>5} {graphrag['evidence_count']:>4}"
        )
    print()

    print("latency (deterministic doubles; seconds)")
    print(format_latency(report["latency"]))
    print(f"peak traced memory: {report['memory_peak_mb']:.2f} MB")
    print()

    print("deletion cleanup")
    print(f"  {report['cleanup']}")
    print()

    print("checks")
    for row in report["checks"]:
        label = "PASS" if row["passed"] else "FAIL"
        print(f"  [{label}] {row['name']}" + (f" - {row['detail']}" if row["detail"] else ""))
    failed = [row for row in report["checks"] if not row["passed"]]
    print(f"\n{len(report['checks']) - len(failed)}/{len(report['checks'])} checks passed")


async def main_async(args: argparse.Namespace) -> int:
    report = await build_report(args.top_k)
    print_report(report)

    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nJSON report written to {destination}")

    failed = [row for row in report["checks"] if not row["passed"]]
    if failed and not args.no_strict:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Retrieval window for the vector comparison (default: 2)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Exit 0 even when a check fails",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
