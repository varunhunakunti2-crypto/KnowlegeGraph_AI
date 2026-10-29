"""Metrics for the GraphRAG evaluation suite.

All metrics are transparent, lexical, and deterministic. They are proxies, not
semantic judges: the docs call out exactly what each one can and cannot prove.
"""
from __future__ import annotations

import re
import statistics
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tests.evaluation.corpus import SourceRef
from tests.evaluation.doubles import content_tokens

_CITATION_RE = re.compile(r"\[chunk:([A-Za-z0-9_-]{1,128})\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

ABSTENTION_MARKERS = (
    "insufficient evidence",
    "does not contain",
    "not enough information",
    "cannot answer",
)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalScore:
    """Rank-aware retrieval score against expected page-level sources."""

    hit_at_k: bool
    recall: float
    precision: float
    reciprocal_rank: float
    retrieved: Tuple[SourceRef, ...]
    expected: Tuple[SourceRef, ...]


def _dedupe(sequence: Iterable[SourceRef]) -> List[SourceRef]:
    seen = set()
    ordered = []
    for item in sequence:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def score_retrieval(
    retrieved: Sequence[SourceRef],
    expected: Sequence[SourceRef],
    k: int,
) -> Optional[RetrievalScore]:
    """Compute hit@k, recall@k, precision@k, and MRR.

    Returns ``None`` when there is no expected source (for example the
    out-of-corpus question), because recall is undefined rather than zero.
    """

    expected_tuple = tuple(_dedupe(expected))
    if not expected_tuple or k <= 0:
        return None

    ordered = _dedupe(retrieved)[:k]
    expected_set = set(expected_tuple)
    hits = [item for item in ordered if item in expected_set]
    hit_at_k = bool(hits)
    reciprocal_rank = 0.0
    for rank, item in enumerate(ordered, start=1):
        if item in expected_set:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalScore(
        hit_at_k=hit_at_k,
        recall=len(set(hits)) / len(expected_set),
        precision=len(hits) / max(1, len(ordered)),
        reciprocal_rank=reciprocal_rank,
        retrieved=tuple(ordered),
        expected=expected_tuple,
    )


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationReport:
    """Citation validity: every citation must point at real retrieved evidence."""

    total: int
    valid: int
    invalid_ids: Tuple[str, ...]
    precision: float
    expected_source_recall: Optional[float]


def _evidence_map(evidence: Sequence[Dict]) -> Dict[str, Dict]:
    mapping: Dict[str, Dict] = {}
    for item in evidence:
        chunk_id = item.get("chunk_id")
        if chunk_id and chunk_id not in mapping:
            mapping[chunk_id] = item
    return mapping


def _evidence_texts(evidence: Sequence[Dict]) -> Dict[str, str]:
    """Concatenate every retrieved fragment for each chunk.

    Graph evidence is split into per-entity/per-relationship fragments, so one
    chunk can appear several times. A claim is supported if any of those
    fragments supports it.
    """

    mapping: Dict[str, List[str]] = {}
    for item in evidence:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue
        mapping.setdefault(str(chunk_id), []).append(str(item.get("text") or ""))
    return {chunk_id: " ".join(texts) for chunk_id, texts in mapping.items()}


def score_citations(
    citations: Sequence[Dict],
    evidence: Sequence[Dict],
    expected_sources: Sequence[SourceRef] = (),
) -> CitationReport:
    """Check that each citation maps to a retrieved chunk with matching provenance."""

    mapping = _evidence_map(evidence)
    valid = 0
    invalid_ids: List[str] = []
    cited_sources = set()

    for citation in citations:
        chunk_id = str(citation.get("chunk_id") or "")
        source = mapping.get(chunk_id)
        if source is None:
            invalid_ids.append(chunk_id or "<missing>")
            continue
        try:
            page_matches = int(citation.get("page_number", -1)) == int(
                source.get("page_number", -2)
            )
        except (TypeError, ValueError):
            page_matches = False
        document_matches = str(citation.get("document_id")) == str(
            source.get("document_id")
        )
        if not (page_matches and document_matches):
            invalid_ids.append(chunk_id)
            continue
        valid += 1
        cited_sources.add(
            SourceRef(
                document_key=str(source.get("document_id")),
                page_number=int(source.get("page_number", 0)),
            )
        )

    expected = set(expected_sources)
    expected_recall = None
    if expected:
        expected_recall = len(cited_sources & expected) / len(expected)

    return CitationReport(
        total=len(citations),
        valid=valid,
        invalid_ids=tuple(invalid_ids),
        precision=valid / len(citations) if citations else 0.0,
        expected_source_recall=expected_recall,
    )


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaithfulnessReport:
    """Lexical support of each cited sentence by the cited evidence."""

    sentences: int
    supported: int
    score: float
    unsupported: Tuple[str, ...] = ()
    abstained: bool = False


def is_abstention(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


def score_faithfulness(
    answer: str,
    citations: Sequence[Dict],
    evidence: Sequence[Dict],
    threshold: float = 0.6,
) -> FaithfulnessReport:
    """Measure whether cited sentences are actually supported by their sources.

    This is a lexical overlap proxy (content-token recall), not an entailment
    model. It reliably catches fabricated claims and invalid citations; it can
    over-reward verbatim extraction and under-reward valid paraphrases.
    """

    if is_abstention(answer):
        return FaithfulnessReport(sentences=0, supported=0, score=1.0, abstained=True)

    support_texts = _evidence_texts(evidence)
    valid_ids = {str(citation.get("chunk_id")) for citation in citations}
    # Models often emit "Sentence. [chunk:id]" with a space. Attach trailing
    # citation markers to the sentence they support so a marker is not counted
    # as its own uncited claim.
    normalized_answer = re.sub(r"\s+(\[chunk:)", r"\1", answer or "")
    sentences = [
        part.strip()
        for part in _SENTENCE_RE.split(normalized_answer)
        if part.strip()
    ]

    supported = 0
    unsupported: List[str] = []
    evaluated = 0

    for sentence in sentences:
        cited = _CITATION_RE.findall(sentence)
        if not cited:
            # A sentence with factual content but no citation is unsupported.
            lexical = content_tokens(sentence)
            if not lexical:
                continue
            evaluated += 1
            unsupported.append(sentence)
            continue

        evaluated += 1
        usable_chunks = [chunk_id for chunk_id in cited if chunk_id in valid_ids]
        support_text = " ".join(
            support_texts.get(chunk_id, "") for chunk_id in usable_chunks
        )
        claim = _CITATION_RE.sub(" ", sentence)
        claim_tokens = set(content_tokens(claim))
        support_tokens = set(content_tokens(support_text))
        if not claim_tokens:
            continue
        ratio = len(claim_tokens & support_tokens) / len(claim_tokens)
        if usable_chunks and ratio >= threshold:
            supported += 1
        else:
            unsupported.append(sentence)

    score = supported / evaluated if evaluated else 1.0
    return FaithfulnessReport(
        sentences=evaluated,
        supported=supported,
        score=score,
        unsupported=tuple(unsupported),
    )


# ---------------------------------------------------------------------------
# Graph accuracy
# ---------------------------------------------------------------------------

Triple = Tuple[str, str, str]


def _normalize_triple(triple: Triple) -> Triple:
    subject, predicate, obj = triple
    return (
        str(subject).strip().lower(),
        str(predicate).strip().upper(),
        str(obj).strip().lower(),
    )


@dataclass(frozen=True)
class GraphScore:
    """Precision/recall/F1 over subject-predicate-object triples."""

    precision: float
    recall: float
    f1: float
    missing: Tuple[Triple, ...] = ()
    extra: Tuple[Triple, ...] = ()
    predicted_count: int = 0
    gold_count: int = 0


def score_graph(predicted: Iterable[Triple], gold: Iterable[Triple]) -> GraphScore:
    predicted_set = {_normalize_triple(triple) for triple in predicted}
    gold_set = {_normalize_triple(triple) for triple in gold}
    if not predicted_set or not gold_set:
        return GraphScore(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            missing=tuple(sorted(gold_set - predicted_set)),
            extra=tuple(sorted(predicted_set - gold_set)),
            predicted_count=len(predicted_set),
            gold_count=len(gold_set),
        )

    true_positives = len(predicted_set & gold_set)
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(gold_set)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return GraphScore(
        precision=precision,
        recall=recall,
        f1=f1,
        missing=tuple(sorted(gold_set - predicted_set)),
        extra=tuple(sorted(predicted_set - gold_set)),
        predicted_count=len(predicted_set),
        gold_count=len(gold_set),
    )


# ---------------------------------------------------------------------------
# Latency and memory
# ---------------------------------------------------------------------------


class LatencyRecorder:
    """Collect wall-clock durations per pipeline stage."""

    def __init__(self):
        self.samples: Dict[str, List[float]] = {}

    @contextmanager
    def time(self, stage: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.samples.setdefault(stage, []).append(time.perf_counter() - started)

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for stage, values in self.samples.items():
            ordered = sorted(values)
            p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
            summary[stage] = {
                "count": float(len(values)),
                "total_s": sum(values),
                "mean_s": statistics.fmean(values),
                "p50_s": statistics.median(values),
                "p95_s": ordered[p95_index],
                "max_s": ordered[-1],
            }
        return summary


class MemoryTracker:
    """Peak Python allocation (tracemalloc) for a block of work.

    This measures in-process allocations only. It does not include native
    allocations in Postgres/Neo4j/Ollama or the full RSS of a server process;
    see docs/EVALUATION.md for the live-suite caveat.
    """

    def __init__(self):
        self.peak_bytes = 0
        self._started = False

    def __enter__(self):
        tracemalloc.start()
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb):
        current, peak = tracemalloc.get_traced_memory()
        self.peak_bytes = peak
        tracemalloc.stop()
        self._started = False
        return False

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)


def format_latency(summary: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{'stage':<14} {'n':>3} {'total_s':>9} {'p50_s':>8} {'p95_s':>8}"]
    for stage, values in sorted(summary.items()):
        lines.append(
            f"{stage:<14} {int(values['count']):>3} {values['total_s']:>9.3f} "
            f"{values['p50_s']:>8.3f} {values['p95_s']:>8.3f}"
        )
    return "\n".join(lines)
