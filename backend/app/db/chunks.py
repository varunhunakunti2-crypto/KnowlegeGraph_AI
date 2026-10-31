"""Chunk database operations with idempotent ingestion."""
import psycopg
from app.core.config import settings
from typing import List, Dict, Optional
import json
from uuid import uuid4


class ChunkDB:
    """PostgreSQL operations for chunks with transaction safety."""
    
    @staticmethod
    def get_connection():
        return psycopg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            connect_timeout=settings.DB_CONNECT_TIMEOUT_SECONDS,
        )
    
    @staticmethod
    def create_chunks_idempotent(document_id: str, chunks: List[Dict], pages: List[Dict] = None) -> int:
        """
        Create chunks idempotently - retrying won't create duplicates.
        Uses transactions and handles partial failures safely.
        """
        conn = ChunkDB.get_connection()
        try:
            with conn.transaction():
                cur = conn.cursor()
                
                # Delete existing chunks for this document (idempotent retry)
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                
                # Insert pages if provided
                if pages:
                    for page in pages:
                        cur.execute("""
                            INSERT INTO document_pages (document_id, page_number, text_content, char_count)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (document_id, page_number) DO UPDATE
                            SET text_content = EXCLUDED.text_content, char_count = EXCLUDED.char_count
                        """, (document_id, page["page_number"], page["text"], len(page["text"])))
                
                # Insert chunks
                count = 0
                for chunk in chunks:
                    cur.execute("""
                        INSERT INTO document_chunks 
                        (document_id, page_number, chunk_index, chunk_text, char_count, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        document_id,
                        chunk["page_number"],
                        chunk["chunk_index"],
                        chunk["chunk_text"],
                        chunk["char_count"],
                        json.dumps({"source": "pdf_extraction"})
                    ))
                    count += 1
                
                cur.close()
            conn.commit()
            return count
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    @staticmethod
    def get_chunks_by_document(document_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Return chunks ordered by index, optionally bounded by ``limit``.

        Graph extraction passes the configured work limit so a large document
        is never materialized in full just to be truncated afterwards.
        """
        query = """
            SELECT chunk_id, document_id, page_number, chunk_index,
                   chunk_text, char_count, created_at
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY chunk_index
        """
        params: list = [document_id]
        if limit is not None:
            bound = max(1, min(int(limit), 100_000))
            query += " LIMIT %s"
            params.append(bound)

        conn = ChunkDB.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        finally:
            conn.close()
        return [{
            "chunk_id": str(row[0]),
            "document_id": str(row[1]),
            "page_number": row[2],
            "chunk_index": row[3],
            "chunk_text": row[4],
            "char_count": row[5],
            "created_at": row[6]
        } for row in rows]
    
    @staticmethod
    def count_chunks(document_id: str) -> int:
        conn = ChunkDB.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_chunks WHERE document_id = %s", (document_id,))
                count = cur.fetchone()[0]
            return count
        finally:
            conn.close()

    @staticmethod
    def delete_chunks_for_document(document_id: str) -> None:
        """Delete chunks and pages for a document.

        Embeddings live on document_chunks, so this also removes vector data.
        The documents row is deleted separately by DocumentDB.
        """
        conn = ChunkDB.get_connection()
        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM document_chunks WHERE document_id = %s",
                        (document_id,),
                    )
                    cur.execute(
                        "DELETE FROM document_pages WHERE document_id = %s",
                        (document_id,),
                    )
        finally:
            conn.close()
