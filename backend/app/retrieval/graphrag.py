"""GraphRAG retrieval combining vector and graph search."""
from app.db.vectors import VectorDB
from app.db.graph import GraphDB
from app.core.config import settings
from app.core.logging import logger
from app.core.security import clamp_int
from typing import List, Dict, Optional
import asyncio


class GraphRAGRetriever:
    """Hybrid retrieval using vector search and graph traversal."""

    def __init__(self):
        self.vector_db = VectorDB()
        self.graph_db = GraphDB()
        self.graph_traversal_depth = clamp_int(
            settings.GRAPH_TRAVERSAL_DEPTH,
            1,
            settings.MAX_GRAPH_TRAVERSAL_DEPTH,
            2,
        )
        self.max_context_tokens = clamp_int(
            settings.MAX_CONTEXT_TOKENS,
            256,
            settings.MODEL_CONTEXT_LENGTH,
            4096,
        )

    async def retrieve(self, question: str, document_id: Optional[str] = None,
                      top_k: int = None) -> Dict:
        """
        Retrieve evidence using both vector and graph search.

        Every retrieval path is bounded by limits from settings. When a
        document_id is supplied, both vector and graph evidence are restricted
        to that document.
        """
        if top_k is None:
            top_k = settings.TOP_K_CHUNKS
        top_k = clamp_int(top_k, 1, settings.MAX_TOP_K, settings.TOP_K_CHUNKS)

        evidence_list = []

        # 1. Vector retrieval
        try:
            vector_results = await self.vector_db.search_similar(
                question, top_k=top_k, document_id=document_id
            )
            for result in vector_results:
                evidence_list.append({
                    "source_type": "vector",
                    "chunk_id": result["chunk_id"],
                    "document_id": result["document_id"],
                    "page_number": result["page_number"],
                    "text": result["text"],
                    "score": result["similarity"] * 0.6,  # Weighting
                    "entity_id": None,
                    "relationship_info": None
                })
        except Exception as e:
            logger.error(f"Vector retrieval failed: {type(e).__name__}")

        # 2. Graph retrieval (synchronous Neo4j driver, keep off the event loop)
        try:
            graph_results = await asyncio.to_thread(
                self._graph_retrieval, question, document_id
            )
            for result in graph_results:
                # Add supporting chunks for graph facts
                evidence_list.append({
                    "source_type": "graph",
                    "chunk_id": result.get("chunk_id"),
                    "document_id": result["document_id"],
                    "page_number": result["page_number"],
                    "text": result["evidence"],
                    "score": result["confidence"] * 0.4,  # Weighting
                    "entity_id": result.get("entity_id"),
                    "relationship_info": result.get("relationship")
                })
        except Exception as e:
            logger.error(f"Graph retrieval failed: {type(e).__name__}")

        # 3. Deduplicate evidence
        evidence_list = self._deduplicate_evidence(evidence_list)

        # 4. Rank evidence
        evidence_list = sorted(evidence_list, key=lambda x: x.get("score", 0), reverse=True)

        # Keep response size and model context bounded even if the databases
        # return more rows than expected.
        evidence_list = self._prepare_evidence(evidence_list)

        # 5. Check if has evidence
        if not evidence_list:
            return {
                "evidence": [],
                "has_evidence": False,
                "context": "",
                "total_tokens_estimate": 0,
                "message": "Insufficient evidence to answer the question"
            }

        # 6. Limit context to fit budget (rough estimate: 1 token ~ 4 chars).
        # Truncating the final item instead of breaking guarantees the context
        # is never empty just because one chunk was oversized.
        context_parts = []
        total_chars = 0
        max_chars = self.max_context_tokens * 4

        for ev in evidence_list:
            header = (
                f"[{ev['source_type']}] Doc:{str(ev['document_id'])[:8]} "
                f"Page:{ev['page_number']}: "
            )
            remaining = max_chars - total_chars
            if remaining <= len(header) + 8:
                break
            body_budget = remaining - len(header) - 2
            body = (ev.get("text") or "")[:body_budget]
            text = f"{header}{body}\n\n"
            context_parts.append(text)
            total_chars += len(text)

        context = "".join(context_parts)

        return {
            "evidence": evidence_list[:settings.MAX_EVIDENCE_ITEMS],
            "has_evidence": True,
            "context": context,
            "total_tokens_estimate": total_chars // 4
        }

    def _prepare_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        """Sanitize and bound evidence before it reaches the LLM or API."""
        prepared = []
        for ev in evidence_list:
            raw_text = ev.get("text")
            if raw_text is None:
                raw_text = ""
            text = str(raw_text).replace("\x00", "")
            text = text[:settings.MAX_EVIDENCE_CHARS]
            ev["text"] = text
            ev["document_id"] = str(ev.get("document_id") or "unknown")
            ev["page_number"] = ev.get("page_number") or 0
            ev["score"] = float(ev.get("score") or 0.0)
            prepared.append(ev)
            if len(prepared) >= settings.MAX_EVIDENCE_ITEMS:
                break
        return prepared

    def _graph_retrieval(self, question: str, document_id: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant entities and relationships via bounded traversal.

        With a document filter, the optional path must end at an entity that is
        also sourced from that document. This prevents a document-scoped chat
        from seeing entity names that only exist in another document.
        """
        driver = self.graph_db.connect()
        results = []
        limit = min(50, settings.MAX_GRAPH_LIMIT)
        depth = self.graph_traversal_depth

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            if document_id:
                query = f"""
                MATCH (e:Entity)-[r:SOURCED_FROM]->(d:Document {{id: $doc_id}})
                OPTIONAL MATCH (e)-[rel:RELATION*1..{depth}]-(e2:Entity)-[:SOURCED_FROM]->(d)
                WHERE e2 IS NULL OR e <> e2
                RETURN e, r, rel, e2
                LIMIT $limit
                """
                result = session.run(query, doc_id=document_id, limit=limit)
            else:
                query = f"""
                MATCH (e:Entity)-[r:SOURCED_FROM]->(d:Document)
                OPTIONAL MATCH (e)-[rel:RELATION*1..{depth}]-(e2:Entity)
                RETURN e, r, rel, e2
                LIMIT $limit
                """
                result = session.run(query, limit=limit)

            for record in result:
                e = record["e"]
                r = record["r"]
                if e and r:
                    evidence_text = r.get("evidence") or ""
                    if not evidence_text:
                        # A relationship without source text is not usable as
                        # evidence and only wastes context budget.
                        continue
                    results.append({
                        "entity_id": e["id"],
                        "document_id": r.get("document_id") or document_id or "unknown",
                        "page_number": r.get("page_number") or 0,
                        "evidence": evidence_text,
                        "confidence": float(e.get("confidence") or 0.5),
                        "chunk_id": r.get("chunk_id")
                    })
                    if len(results) >= settings.MAX_EVIDENCE_ITEMS:
                        break

        return results

    def _deduplicate_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        """Merge duplicate evidence."""
        seen = {}
        for ev in evidence_list:
            text = str(ev.get("text") or "")
            key = (ev.get("document_id"), ev.get("page_number"), text[:100])
            if key not in seen or ev.get("score", 0) > seen[key].get("score", 0):
                seen[key] = ev
        return list(seen.values())


graphrag_retriever = GraphRAGRetriever()
