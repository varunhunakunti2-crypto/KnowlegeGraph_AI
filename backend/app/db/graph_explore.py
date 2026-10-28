"""Graph exploration queries."""
from neo4j import GraphDatabase

from app.core.config import settings
from app.core.security import clamp_int
from typing import List, Dict, Optional


class GraphExplorer:
    """Graph exploration with bounded queries.

    All user-controlled values are passed as Cypher parameters. The only values
    interpolated into query text are integers that have been clamped to a
    constant-safe range.
    """

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

    def search_entities(self, query: str, node_type: Optional[str] = None,
                       document_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Search entities by name."""
        limit = clamp_int(limit, 1, min(100, settings.MAX_GRAPH_LIMIT), 20)
        driver = self.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            where_clauses = ["(toLower(e.name) CONTAINS toLower($search_term) OR any(a IN e.aliases WHERE toLower(a) CONTAINS toLower($search_term)))"]
            # NOTE: the parameter must not be named "query": neo4j's run() takes a
            # positional argument called ``query`` and ``run(cypher, **params)``
            # would raise TypeError for a parameter of the same name.
            params = {"search_term": query[:200], "limit": limit}

            if node_type:
                where_clauses.append("e.type = $node_type")
                params["node_type"] = node_type

            if document_id:
                where_clauses.append("EXISTS((e)-[:SOURCED_FROM]->(:Document {id: $doc_id}))")
                params["doc_id"] = document_id

            cypher = f"""
            MATCH (e:Entity)
            WHERE {' AND '.join(where_clauses)}
            RETURN e.id as id, e.name as label, e.type as type
            LIMIT $limit
            """
            result = session.run(cypher, **params)
            entities = []
            for record in result:
                entities.append({
                    "id": record["id"],
                    "label": record["label"],
                    "type": record["type"],
                    "document_count": 0
                })
            return entities

    def get_entity_neighbors(self, entity_id: str, depth: int = 1,
                            node_type: Optional[str] = None,
                            rel_type: Optional[str] = None,
                            limit: int = 50,
                            document_id: Optional[str] = None) -> Dict:
        """Get entity and its neighbors with bounded traversal."""
        depth = clamp_int(depth, 1, settings.MAX_GRAPH_TRAVERSAL_DEPTH, 1)
        limit = clamp_int(limit, 1, min(100, settings.MAX_GRAPH_LIMIT), 50)

        driver = self.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            filters = []
            params = {"entity_id": entity_id, "limit": limit}

            if document_id:
                # Both the start node and every traversed relationship must
                # belong to the requested document. ``rel`` and ``neighbor``
                # are NULL for an isolated entity, so the NULL branch keeps the
                # entity itself visible instead of turning it into a 404.
                filters.append(
                    "(rel IS NULL OR (all(r in rel WHERE r.document_id = $doc_id) "
                    "AND EXISTS((neighbor)-[:SOURCED_FROM]->(:Document {id: $doc_id}))))"
                )
                params["doc_id"] = document_id

            if node_type:
                filters.append("(neighbor IS NULL OR neighbor.type = $node_type)")
                params["node_type"] = node_type

            if rel_type:
                filters.append(
                    "(rel IS NULL OR all(r in rel WHERE r.type = $rel_type))"
                )
                params["rel_type"] = rel_type

            filter_clause = ("AND " + " AND ".join(filters)) if filters else ""
            source_match = "MATCH (e:Entity {id: $entity_id})"
            if document_id:
                source_match += "\n            MATCH (e)-[:SOURCED_FROM]->(:Document {id: $doc_id})"

            # depth is an int clamped above, so it is safe to interpolate.
            cypher = f"""
            {source_match}
            OPTIONAL MATCH path = (e)-[rel:RELATION*1..{depth}]-(neighbor:Entity)
            WHERE (neighbor IS NULL OR neighbor <> e)
            {filter_clause}
            RETURN e, rel, neighbor
            LIMIT $limit
            """
            result = session.run(cypher, **params)
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
                neighbor = record["neighbor"]
                if neighbor:
                    nodes[neighbor["id"]] = {
                        "id": neighbor["id"],
                        "label": neighbor["name"],
                        "type": neighbor["type"],
                        "aliases": neighbor.get("aliases", [])
                    }
                rels = record["rel"]
                if rels:
                    for rel in rels:
                        if hasattr(rel, 'type') and rel.type == 'SOURCED_FROM':
                            continue
                        edges.append({
                            "id": f"e{hash(str(rel)) % 10000}",
                            "source": e["id"] if e else "",
                            "target": neighbor["id"] if neighbor else "",
                            "label": rel.get("type", "RELATED") if hasattr(rel, 'get') else "RELATED",
                            "confidence": rel.get("confidence") if hasattr(rel, 'get') else None,
                            "evidence": []
                        })

            return {
                "nodes": list(nodes.values())[:settings.MAX_GRAPH_LIMIT],
                "edges": edges[:50],  # Limit edges
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            }

    def get_entity_evidence(self, entity_id: str, limit: int = 20,
                            document_id: Optional[str] = None) -> List[Dict]:
        """Get source evidence for an entity, optionally scoped to a document."""
        limit = clamp_int(limit, 1, 100, 20)
        driver = self.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            cypher = """
            MATCH (e:Entity {id: $entity_id})-[r:SOURCED_FROM]->(d:Document)
            WHERE $doc_id IS NULL OR d.id = $doc_id
            RETURN r.evidence as evidence, r.chunk_id as chunk_id,
                   r.page_number as page_number, d.id as document_id
            LIMIT $limit
            """
            result = session.run(
                cypher, entity_id=entity_id, limit=limit, doc_id=document_id
            )
            evidence = []
            for record in result:
                evidence.append({
                    "evidence": record["evidence"],
                    "chunk_id": record["chunk_id"],
                    "page_number": record["page_number"],
                    "document_id": record["document_id"]
                })
            return evidence

    def get_edge_evidence(self, source_id: str, target_id: str, limit: int = 10,
                          document_id: Optional[str] = None) -> List[Dict]:
        """Get source evidence for an edge, optionally scoped to a document."""
        limit = clamp_int(limit, 1, 20, 10)
        driver = self.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            cypher = """
            MATCH (s:Entity {id: $source_id})-[r:RELATION]-(t:Entity {id: $target_id})
            WHERE r.evidence IS NOT NULL
              AND ($doc_id IS NULL OR r.document_id = $doc_id)
            RETURN r.evidence as evidence, r.chunk_id as chunk_id,
                   r.page_number as page_number, r.document_id as document_id,
                   r.type as relationship_type
            LIMIT $limit
            """
            result = session.run(
                cypher,
                source_id=source_id,
                target_id=target_id,
                limit=limit,
                doc_id=document_id,
            )
            evidence = []
            for record in result:
                evidence.append({
                    "evidence": record["evidence"],
                    "chunk_id": record["chunk_id"],
                    "page_number": record["page_number"],
                    "document_id": record["document_id"],
                    "relationship_type": record["relationship_type"]
                })
            return evidence

    def get_document_graph(self, document_id: str, limit_nodes: int = 100) -> Dict:
        """Get graph for specific document."""
        limit_nodes = clamp_int(
            limit_nodes, 1, min(200, settings.MAX_GRAPH_LIMIT), 100
        )
        driver = self.connect()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            cypher = """
            MATCH (e:Entity)-[:SOURCED_FROM]->(d:Document {id: $doc_id})
            OPTIONAL MATCH (e)-[r:RELATION]-(e2:Entity)-[:SOURCED_FROM]->(d)
            WHERE e2 IS NULL OR e <> e2
            RETURN e, r, e2
            LIMIT $limit
            """
            result = session.run(cypher, doc_id=document_id, limit=limit_nodes)
            nodes = {}
            edges = []
            for record in result:
                e = record["e"]
                if e and e["id"] not in nodes:
                    nodes[e["id"]] = {"id": e["id"], "label": e["name"], "type": e["type"], "aliases": e.get("aliases", [])}
                e2 = record["e2"]
                if e2 and e2["id"] not in nodes:
                    nodes[e2["id"]] = {"id": e2["id"], "label": e2["name"], "type": e2["type"], "aliases": e2.get("aliases", [])}
                r = record["r"]
                if r and e and e2:
                    edges.append({"id": f"e{hash(str(r))}", "source": e["id"], "target": e2["id"],
                                "label": r.get("type", "RELATED"), "confidence": r.get("confidence")})
            return {"nodes": list(nodes.values())[:limit_nodes], "edges": edges[:limit_nodes]}


graph_explorer = GraphExplorer()
