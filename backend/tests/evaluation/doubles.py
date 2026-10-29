"""Deterministic test doubles for the GraphRAG evaluation pipeline.

The doubles intentionally sit at the same boundaries as the production
components (embedding service, vector store, Neo4j graph store, Ollama LLM) so
the real ingestion, retrieval, generation, citation-validation, and cleanup code
runs unchanged.

They are *scripted*, not intelligent: the answer double can only answer when the
retrieved evidence actually contains the expected answer phrase, and it cites
only real evidence chunk IDs. That makes retrieval failures visible in the
metrics and keeps the suite deterministic. Model quality itself is evaluated by
the optional live suite, not here.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ANSWER_EVIDENCE_RE = re.compile(
    r'<evidence id="([A-Za-z0-9_-]{1,128})" doc="([^"]*)" page="(\d+)">\n(.*?)\n</evidence>',
    re.DOTALL,
)
_QUESTION_RE = re.compile(r"^Question: (.*)$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

ABSTENTION_TEXT = (
    "The provided evidence does not contain enough information to answer this question."
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "then", "there", "these", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens with simple possessive handling."""

    normalized = str(text or "").lower().replace("'s", "")
    return _TOKEN_RE.findall(normalized)


def content_tokens(text: str) -> List[str]:
    """Tokens that carry meaning for lexical overlap metrics."""

    return [token for token in tokenize(text) if token not in STOPWORDS]


class DeterministicEmbedder:
    """TF-IDF hashed embedder with a fixed dimension.

    It behaves like a small semantic embedder for this closed corpus: related
    text shares vocabulary and therefore ranks highly, while unrelated questions
    have zero similarity. The point is reproducibility for CI, not fidelity to
    nomic-embed-text.
    """

    def __init__(self, corpus_texts: Sequence[str], dimension: int = 768):
        document_tokens = [set(tokenize(text)) for text in corpus_texts]
        document_count = max(1, len(document_tokens))
        document_frequency: Counter[str] = Counter()
        for tokens in document_tokens:
            document_frequency.update(tokens)
        self.idf = {
            token: math.log((document_count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self.vocabulary = set(self.idf)
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        counts = Counter(tokenize(text))
        for token, frequency in counts.items():
            if token not in self.vocabulary:
                continue
            weight = (1.0 + math.log(frequency)) * self.idf[token]
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            vector[bucket % self.dimension] += weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity for already-normalized vectors."""

    return sum(a * b for a, b in zip(left, right))


class InMemoryVectorStore:
    """Async stand-in for ``VectorDB`` backed by a list of chunks."""

    def __init__(self, embedder: DeterministicEmbedder):
        self.embedder = embedder
        self.rows: List[Dict[str, Any]] = []

    def index(
        self,
        *,
        chunk_id: str,
        document_id: str,
        page_number: int,
        chunk_index: int,
        text: str,
        embedding: Sequence[float],
    ) -> None:
        self.rows = [row for row in self.rows if row["chunk_id"] != chunk_id]
        self.rows.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": text,
                "char_count": len(text),
                "embedding": list(embedding),
            }
        )

    async def search_similar(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed(query)
        candidates = [
            row
            for row in self.rows
            if document_id is None or row["document_id"] == document_id
        ]
        scored = [
            (
                cosine(query_embedding, row["embedding"]),
                row,
            )
            for row in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        limit = len(scored) if top_k is None else max(0, min(int(top_k), len(scored)))
        results = []
        for similarity, row in scored[:limit]:
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "page_number": row["page_number"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "char_count": row["char_count"],
                    "similarity": similarity,
                }
            )
        return results

    def delete_document(self, document_id: str) -> int:
        before = len(self.rows)
        self.rows = [row for row in self.rows if row["document_id"] != document_id]
        return before - len(self.rows)

    def count(self, document_id: Optional[str] = None) -> int:
        if document_id is None:
            return len(self.rows)
        return sum(1 for row in self.rows if row["document_id"] == document_id)

    def has_chunk(self, chunk_id: str) -> bool:
        return any(row["chunk_id"] == chunk_id for row in self.rows)


class InMemoryGraphStore:
    """Stand-in for ``GraphDB`` with the same provenance semantics.

    Mirrors the production filtering rules that matter to the evaluator:
    entities and relationships are scoped by document, relationships carry the
    source chunk/page/evidence, and deletion only removes an entity once no
    document sources it anymore.
    """

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.provenance: List[Dict[str, Any]] = []
        self.relationships: List[Dict[str, Any]] = []

    @staticmethod
    def _entity_id(name: str, entity_type: str) -> str:
        normalized = str(name).strip().lower()
        return hashlib.sha256(f"{entity_type}:{normalized}".encode("utf-8")).hexdigest()[:16]

    def store_extraction(self, document_id: str, extraction_result) -> Dict[str, int]:
        entities_stored = 0
        relationships_stored = 0

        for entity in extraction_result.entities:
            entity_id = self._entity_id(entity.name, entity.type)
            if entity_id not in self.entities:
                self.entities[entity_id] = {
                    "id": entity_id,
                    "name": entity.name,
                    "type": entity.type,
                    "aliases": list(entity.aliases),
                    "confidence": entity.confidence,
                }
                entities_stored += 1
            provenance_key = (
                entity_id,
                document_id,
                entity.source_chunk_id,
                entity.page_number,
            )
            if not any(
                (row["entity_id"], row["document_id"], row["chunk_id"], row["page_number"])
                == provenance_key
                for row in self.provenance
            ):
                self.provenance.append(
                    {
                        "entity_id": entity_id,
                        "document_id": document_id,
                        "chunk_id": entity.source_chunk_id,
                        "page_number": entity.page_number,
                        "evidence": entity.evidence,
                        "confidence": entity.confidence,
                    }
                )

        for relationship in extraction_result.relationships:
            key = (
                document_id,
                relationship.subject.strip().lower(),
                relationship.predicate,
                relationship.object.strip().lower(),
            )
            if any(
                (
                    row["document_id"],
                    row["subject"].strip().lower(),
                    row["predicate"],
                    row["object"].strip().lower(),
                )
                == key
                for row in self.relationships
            ):
                continue
            self.relationships.append(
                {
                    "document_id": document_id,
                    "subject": relationship.subject,
                    "predicate": relationship.predicate,
                    "object": relationship.object,
                    "confidence": relationship.confidence,
                    "chunk_id": relationship.source_chunk_id,
                    "page_number": relationship.page_number,
                    "evidence": relationship.evidence,
                }
            )
            relationships_stored += 1

        return {
            "entities_stored": entities_stored,
            "relationships_stored": relationships_stored,
        }

    def _entity_id_by_name(self, name: str, document_id: str) -> Optional[str]:
        normalized = str(name).strip().lower()
        for entity_id, entity in self.entities.items():
            if entity["name"].strip().lower() != normalized:
                continue
            if any(
                row["entity_id"] == entity_id and row["document_id"] == document_id
                for row in self.provenance
            ):
                return entity_id
        return None

    def retrieve(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return question-independent graph evidence, mirroring ``_graph_retrieval``.

        Production GraphRAG currently traverses every entity sourced from the
        requested document(s); it does not match entities against the question.
        The double models the same behaviour so precision/recall findings are
        representative.
        """

        results: List[Dict[str, Any]] = []
        relationships = [
            row
            for row in self.relationships
            if document_id is None or row["document_id"] == document_id
        ]
        for relationship in relationships:
            entity_id = self._entity_id_by_name(relationship["subject"], relationship["document_id"])
            results.append(
                {
                    "entity_id": entity_id,
                    "document_id": relationship["document_id"],
                    "page_number": relationship["page_number"],
                    "evidence": relationship["evidence"],
                    "confidence": relationship["confidence"],
                    "chunk_id": relationship["chunk_id"],
                }
            )

        for row in self.provenance:
            if document_id is not None and row["document_id"] != document_id:
                continue
            results.append(
                {
                    "entity_id": row["entity_id"],
                    "document_id": row["document_id"],
                    "page_number": row["page_number"],
                    "evidence": row["evidence"],
                    "confidence": row["confidence"],
                    "chunk_id": row["chunk_id"],
                }
            )
        return results[:50]

    def get_graph(self, document_id: Optional[str] = None, limit: int = 100) -> Dict[str, List[Dict]]:
        entity_ids = {
            row["entity_id"]
            for row in self.provenance
            if document_id is None or row["document_id"] == document_id
        }
        nodes = [
            {
                "id": entity_id,
                "label": self.entities[entity_id]["name"],
                "type": self.entities[entity_id]["type"],
                "aliases": list(self.entities[entity_id]["aliases"]),
            }
            for entity_id in sorted(entity_ids)
        ][:limit]

        edges: List[Dict[str, Any]] = []
        for relationship in self.relationships:
            if document_id is not None and relationship["document_id"] != document_id:
                continue
            source = self._entity_id_by_name(relationship["subject"], relationship["document_id"])
            target = self._entity_id_by_name(relationship["object"], relationship["document_id"])
            if not source or not target or source == target:
                continue
            edges.append(
                {
                    "id": f"{source}-{target}",
                    "source": source,
                    "target": target,
                    "label": relationship["predicate"],
                    "confidence": relationship["confidence"],
                }
            )
        return {"nodes": nodes, "edges": edges[:limit]}

    def delete_document_graph(self, document_id: str) -> None:
        self.provenance = [
            row for row in self.provenance if row["document_id"] != document_id
        ]
        self.relationships = [
            row for row in self.relationships if row["document_id"] != document_id
        ]
        sourced = {row["entity_id"] for row in self.provenance}
        self.entities = {
            entity_id: entity
            for entity_id, entity in self.entities.items()
            if entity_id in sourced
        }

    def counts(self, document_id: Optional[str] = None) -> Dict[str, int]:
        provenance = [
            row
            for row in self.provenance
            if document_id is None or row["document_id"] == document_id
        ]
        relationships = [
            row
            for row in self.relationships
            if document_id is None or row["document_id"] == document_id
        ]
        return {
            "entities": len({row["entity_id"] for row in provenance}),
            "relationships": len(relationships),
            "provenance": len(provenance),
        }

    def relation_triples(self) -> List[tuple[str, str, str]]:
        return [
            (
                row["subject"].strip().lower(),
                row["predicate"],
                row["object"].strip().lower(),
            )
            for row in self.relationships
        ]


class FixedResponseLLM:
    """Returns one scripted response; used to drive graph extraction."""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        return self.response


class ScriptedAnswerLLM:
    """Deterministic extractive answerer used to exercise generation wiring.

    Modes:
    - ``correct``: cite the evidence sentence that contains the expected answer
      phrase, or abstain if no retrieved evidence contains it.
    - ``invalid_citation``: emit a citation ID that is not in the evidence.
    - ``unsupported_claim``: cite a real chunk but state a claim absent from it.
    """

    def __init__(
        self,
        answer_keywords: Dict[str, Sequence[str]],
        mode: str = "correct",
    ):
        self.answer_keywords = {
            question.strip().lower(): tuple(keywords)
            for question, keywords in answer_keywords.items()
        }
        self.mode = mode

    @staticmethod
    def _parse_evidence(prompt: str) -> List[Dict[str, Any]]:
        evidence = []
        for chunk_id, document_id, page_number, text in _ANSWER_EVIDENCE_RE.findall(prompt):
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page_number": int(page_number),
                    "text": text,
                }
            )
        return evidence

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(text) if sentence.strip()]

    def _find_support(
        self,
        keywords: Sequence[str],
        evidence: Sequence[Dict[str, Any]],
    ) -> Optional[tuple[Dict[str, Any], str]]:
        for keyword in keywords:
            needle = keyword.lower()
            for item in evidence:
                text = item["text"]
                if needle not in text.lower():
                    continue
                for sentence in self._sentences(text):
                    if needle in sentence.lower():
                        return item, sentence
                return item, text[:300]
        return None

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        question_match = _QUESTION_RE.search(prompt or "")
        question = (question_match.group(1) if question_match else "").strip().lower()
        evidence = self._parse_evidence(prompt or "")

        if self.mode == "invalid_citation":
            return "Ada Lovelace worked with Charles Babbage on the Analytical Engine. [chunk:missing-chunk-id]"

        if self.mode == "unsupported_claim":
            chunk_id = evidence[0]["chunk_id"] if evidence else "missing-chunk-id"
            return f"Napoleon Bonaparte conquered Russia in 1812. [chunk:{chunk_id}]"

        keywords = self.answer_keywords.get(question, ())
        support = self._find_support(keywords, evidence) if keywords else None
        if support is None:
            return ABSTENTION_TEXT

        item, sentence = support
        return f"{sentence} [chunk:{item['chunk_id']}]"


def iter_evidence_texts(evidence: Iterable[Dict[str, Any]]) -> Iterable[str]:
    for item in evidence:
        yield str(item.get("text") or "")
