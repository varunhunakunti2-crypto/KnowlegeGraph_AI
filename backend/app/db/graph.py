"""Neo4j graph operations."""
from neo4j import GraphDatabase
from app.core.config import settings
from app.core.logging import logger
from typing import List, Dict
import hashlib
import time


class GraphDB:
    """Neo4j graph database operations."""
    
    def __init__(self):
        self.driver = None
    
    def connect(self):
        if not self.driver:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                connection_timeout=settings.DB_CONNECT_TIMEOUT_SECONDS,
                max_connection_lifetime=3600,
                max_transaction_retry_time=15,
            )
        return self.driver
    
    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name."""
        return name.strip().lower()
    
    def _create_entity_id(self, name: str, entity_type: str) -> str:
        """Create stable entity ID."""
        normalized = self._normalize_name(name)
        return hashlib.sha256(f"{entity_type}:{normalized}".encode()).hexdigest()[:16]
    
    def store_extraction(self, document_id: str, extraction_result) -> Dict:
        """Store entities and relationships idempotently."""
        driver = self.connect()
        entities_stored = 0
        rels_stored = 0
        
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            # Store entities
            for entity in extraction_result.entities:
                entity_id = self._create_entity_id(entity.name, entity.type)
                try:
                    session.execute_write(
                        self._create_entity,
                        entity_id,
                        entity.name,
                        entity.type,
                        entity.aliases,
                        entity.confidence,
                        entity.source_chunk_id,
                        entity.source_document_id,
                        entity.page_number,
                        entity.evidence
                    )
                    entities_stored += 1
                except Exception as e:
                    logger.error(f"Failed to store entity: {type(e).__name__}")
            
            # Store relationships
            for rel in extraction_result.relationships:
                try:
                    created = session.execute_write(
                        self._create_relationship,
                        rel.subject,
                        rel.predicate,
                        rel.object,
                        rel.confidence,
                        rel.source_chunk_id,
                        rel.source_document_id,
                        rel.page_number,
                        rel.evidence
                    )
                    if created:
                        rels_stored += 1
                except Exception as e:
                    logger.error(f"Failed to store relationship: {type(e).__name__}")
        
        return {"entities_stored": entities_stored, "relationships_stored": rels_stored}
    
    @staticmethod
    def _create_entity(tx, entity_id, name, entity_type, aliases, confidence, 
                       chunk_id, doc_id, page_num, evidence):
        query = """
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET 
            e.name = $name,
            e.type = $type,
            e.aliases = $aliases,
            e.created_at = timestamp()
        SET e.confidence = $confidence
        WITH e
        MERGE (d:Document {id: $doc_id})
        MERGE (e)-[r:SOURCED_FROM]->(d)
        SET r.chunk_id = $chunk_id,
            r.document_id = $doc_id,
            r.page_number = $page_num,
            r.evidence = $evidence
        """
        tx.run(query, entity_id=entity_id, name=name, type=entity_type, 
               aliases=aliases, confidence=confidence, doc_id=doc_id,
               chunk_id=chunk_id, page_num=page_num, evidence=evidence)
    
    @staticmethod
    def _create_relationship(tx, subject, predicate, obj, confidence,
                            chunk_id, doc_id, page_num, evidence):
        # Both endpoints must be entities sourced from the same document.
        # Without this scope a relationship could silently connect an entity
        # from one document to an unrelated entity in another document.
        # The return value tells the caller whether an edge was actually
        # created: model output can name an endpoint it never extracted, and
        # counting that as a stored relationship would be misleading.
        query = """
        MATCH (s:Entity)-[:SOURCED_FROM]->(:Document {id: $doc_id})
        MATCH (o:Entity)-[:SOURCED_FROM]->(:Document {id: $doc_id})
        WHERE (toLower(s.name) = toLower($subject)
               OR $subject IN [a IN s.aliases | toLower(a)])
          AND (toLower(o.name) = toLower($obj)
               OR $obj IN [a IN o.aliases | toLower(a)])
        WITH s, o LIMIT 1
        MERGE (s)-[r:RELATION {type: $predicate}]->(o)
        ON CREATE SET r.confidence = $confidence, r.created_at = timestamp()
        SET r.chunk_id = $chunk_id,
            r.document_id = $doc_id,
            r.page_number = $page_num,
            r.evidence = $evidence
        RETURN count(r) AS created
        """
        record = tx.run(query, subject=subject, predicate=predicate, obj=obj,
                        confidence=confidence, chunk_id=chunk_id, doc_id=doc_id,
                        page_num=page_num, evidence=evidence).single()
        return bool(record and record["created"])
    
    def get_graph(self, document_id: str = None, limit: int = 100) -> Dict:
        """Get nodes and edges for visualization.

        For a single-document view both endpoints of every relationship must be
        sourced from that document, so the graph cannot reveal names from other
        documents.
        """
        limit = max(1, min(int(limit), settings.MAX_GRAPH_LIMIT))
        driver = self.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            if document_id:
                result = session.run("""
                    MATCH (e:Entity)-[r:SOURCED_FROM]->(d:Document {id: $doc_id})
                    OPTIONAL MATCH (e)-[rel:RELATION]->(e2:Entity)-[:SOURCED_FROM]->(d)
                    WHERE e2 IS NULL OR e <> e2
                    RETURN e, rel, e2 LIMIT $limit
                """, doc_id=document_id, limit=limit)
            else:
                # Relationships are traversed outgoing-only so each edge is
                # returned once. An undirected pattern would emit both the
                # forward and reverse copy of every relationship.
                result = session.run("""
                    MATCH (e:Entity)
                    OPTIONAL MATCH (e)-[rel:RELATION]->(e2:Entity)
                    RETURN e, rel, e2 LIMIT $limit
                """, limit=limit)
            
            nodes = {}
            edges = []
            for record in result:
                e = record["e"]
                if e:
                    nodes[e["id"]] = {
                        "id": e["id"],
                        "label": e["name"],
                        "type": e["type"],
                        "aliases": e.get("aliases", [])
                    }
                e2 = record["e2"]
                if e2 and e2["id"] not in nodes:
                    nodes[e2["id"]] = {
                        "id": e2["id"],
                        "label": e2["name"],
                        "type": e2["type"],
                        "aliases": e2.get("aliases", [])
                    }
                rel = record["rel"]
                if rel and e and e2:
                    edges.append({
                        "id": f"{e['id']}-{e2['id']}",
                        "source": e["id"],
                        "target": e2["id"],
                        "label": rel.get("type", "RELATED"),
                        "confidence": rel.get("confidence")
                    })
            
            return {"nodes": list(nodes.values()), "edges": edges}
    
    def delete_document_graph(self, document_id: str, retries: int = 2):
        """Delete all graph evidence for a document.

        Deletion is idempotent, so transient Neo4j failures are retried with a
        short exponential backoff. Callers still get the final exception so the
        API can report that graph cleanup was incomplete.
        """
        attempts = max(1, int(retries) + 1)
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                driver = self.connect()
                with driver.session(database=settings.NEO4J_DATABASE) as session:
                    session.execute_write(self._delete_document, document_id)
                return
            except Exception as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                delay = min(0.5 * (2 ** attempt), 2.0)
                logger.warning(
                    f"Graph deletion attempt {attempt + 1}/{attempts} failed: "
                    f"{type(exc).__name__}; retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        assert last_error is not None
        raise last_error
    
    @staticmethod
    def _delete_document(tx, document_id):
        """Remove all graph evidence owned by one document.

        Relationship evidence is keyed by document_id, so it must be removed
        explicitly even when the endpoints are shared with another document.
        Entities are only deleted once no document sources them anymore.
        """
        tx.run(
            """
            MATCH (d:Document {id: $doc_id})
            OPTIONAL MATCH (e:Entity)-[r:SOURCED_FROM]->(d)
            DELETE r
            """,
            doc_id=document_id,
        )
        tx.run(
            """
            MATCH ()-[r:RELATION]-()
            WHERE r.document_id = $doc_id
            DELETE r
            """,
            doc_id=document_id,
        )
        tx.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)-[:SOURCED_FROM]->()
            DETACH DELETE e
            """
        )
        tx.run(
            """
            MATCH (d:Document {id: $doc_id})
            DETACH DELETE d
            """,
            doc_id=document_id,
        )


graph_db = GraphDB()
