#!/usr/bin/env python
r"""Reconcile Neo4j graph evidence with PostgreSQL documents.

Safe by default: without ``--apply`` this only reports what would be removed.

Orphan document graphs are Neo4j ``Document`` nodes whose id no longer exists
in PostgreSQL (for example, a document deleted while Neo4j was down). Legacy
``RELATION`` rows created before relationship evidence stored ``document_id``
are also reported. Orphan entities have no remaining ``SOURCED_FROM`` edge.

Usage (from backend/):

    .\.venv\Scripts\python.exe scripts\cleanup_graph.py
    .\.venv\Scripts\python.exe scripts\cleanup_graph.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import psycopg

from app.core.config import settings
from app.db.graph import graph_db


def postgres_document_ids() -> set[str]:
    with psycopg.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=settings.DB_CONNECT_TIMEOUT_SECONDS,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT document_id::text FROM documents")
            return {row[0] for row in cursor.fetchall()}


def graph_state(postgres_ids: set[str]) -> dict:
    driver = graph_db.connect()
    driver.verify_connectivity()
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        graph_ids = [
            record["id"]
            for record in session.run(
                "MATCH (d:Document) WHERE d.id IS NOT NULL RETURN d.id AS id"
            )
        ]
        orphan_documents = sorted(
            document_id for document_id in graph_ids if document_id not in postgres_ids
        )
        legacy_relationships = session.run(
            "MATCH ()-[r:RELATION]->() WHERE r.document_id IS NULL "
            "RETURN count(r) AS n"
        ).single()["n"]
        orphan_entities = session.run(
            "MATCH (e:Entity) WHERE NOT (e)-[:SOURCED_FROM]->() "
            "RETURN count(e) AS n"
        ).single()["n"]

    return {
        "graph_documents": len(graph_ids),
        "orphan_documents": orphan_documents,
        "legacy_relationships": legacy_relationships,
        "orphan_entities": orphan_entities,
    }


def apply_cleanup(orphan_documents: list[str]) -> dict:
    deleted = {"documents": 0, "legacy_relationships": 0, "entities": 0}

    for document_id in orphan_documents:
        graph_db.delete_document_graph(document_id)
        deleted["documents"] += 1

    driver = graph_db.connect()
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        summary = session.run(
            "MATCH ()-[r:RELATION]->() WHERE r.document_id IS NULL DELETE r"
        ).consume()
        deleted["legacy_relationships"] = summary.counters.relationships_deleted

        summary = session.run(
            "MATCH (e:Entity) WHERE NOT (e)-[:SOURCED_FROM]->() DETACH DELETE e"
        ).consume()
        deleted["entities"] = summary.counters.nodes_deleted

    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete reported graph data. Without this flag the script is read-only.",
    )
    args = parser.parse_args()

    postgres_ids = postgres_document_ids()
    state = graph_state(postgres_ids)

    print(f"PostgreSQL documents: {len(postgres_ids)}")
    print(f"Neo4j Document nodes: {state['graph_documents']}")
    print(f"Orphan graph documents: {len(state['orphan_documents'])}")
    for document_id in state["orphan_documents"]:
        print(f"  - {document_id}")
    print(f"Legacy RELATION rows without document_id: {state['legacy_relationships']}")
    print(f"Orphan Entity nodes: {state['orphan_entities']}")

    if not args.apply:
        print("\nDry run: no changes made. Re-run with --apply to remove these.")
        return 0

    deleted = apply_cleanup(state["orphan_documents"])
    print(
        "\nRemoved "
        f"{deleted['documents']} orphan graph document(s), "
        f"{deleted['legacy_relationships']} legacy relationship(s), "
        f"{deleted['entities']} orphan entity/entities."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
