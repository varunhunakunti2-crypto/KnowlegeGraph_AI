"""Vector operations with pgvector."""
import psycopg
from app.core.config import settings
from app.services.embeddings import embedding_service
from app.core.logging import logger
from app.core.security import clamp_int
from app.core import memory
from typing import List, Dict, Optional
from pgvector.psycopg import register_vector


class VectorDB:
    """PostgreSQL pgvector operations."""
    
    @staticmethod
    def get_connection():
        conn = psycopg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            connect_timeout=settings.DB_CONNECT_TIMEOUT_SECONDS,
        )
        register_vector(conn)
        return conn
    
    @staticmethod
    async def verify_embedding_dimensions() -> int:
        """Verify actual embedding dimensions from model."""
        try:
            test_embedding = await embedding_service.embed_text("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to verify embedding dimensions: {type(e).__name__}")
            raise
    
    @staticmethod
    async def index_document_chunks(document_id: str) -> Dict:
        """Index unindexed chunks in small, independently committed batches.

        Resumable by design:

        - only ``chunk_index > last`` rows are fetched, so a document is never
          loaded into memory in full;
        - every successful batch is committed before the next one starts;
        - a failed batch is marked ``failed`` and retried on the next call,
          because only ``indexed`` rows are skipped;
        - indexing stops after ``MAX_INDEX_BATCH_FAILURES`` consecutive errors
          instead of hammering a broken local model.
        """
        conn = VectorDB.get_connection()
        conn.autocommit = True
        indexed_count = 0
        failed_count = 0
        processed_count = 0
        consecutive_failures = 0
        last_index = -1
        stop_reason: Optional[str] = None

        batch_size = clamp_int(
            settings.INDEX_BATCH_SIZE, 1, max(1, settings.BATCH_SIZE), 8
        )
        max_failures = max(1, settings.MAX_INDEX_BATCH_FAILURES)

        try:
            while True:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT chunk_id, chunk_text, chunk_index
                        FROM document_chunks
                        WHERE document_id = %s
                          AND chunk_index > %s
                          AND (embedding_status IS NULL OR embedding_status != 'indexed')
                        ORDER BY chunk_index
                        LIMIT %s
                    """, (document_id, last_index, batch_size))
                    rows = cur.fetchall()

                if not rows:
                    break

                chunk_ids = [row[0] for row in rows]
                texts = [row[1] for row in rows]
                last_index = rows[-1][2]
                processed_count += len(rows)

                try:
                    embeddings = await embedding_service.embed_batch(texts)
                    if len(embeddings) != len(rows):
                        raise ValueError("Embedding batch size mismatch")

                    for embedding in embeddings:
                        if len(embedding) != settings.EMBEDDING_DIMENSION:
                            raise ValueError(
                                "Embedding dimension does not match "
                                f"EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION}"
                            )

                    # Commit this batch before moving on, so a crash or OOM in a
                    # later batch cannot discard completed work.
                    with conn.transaction():
                        with conn.cursor() as cur:
                            for chunk_id, embedding in zip(chunk_ids, embeddings):
                                cur.execute("""
                                    UPDATE document_chunks
                                    SET embedding = %s,
                                        embedding_model = %s,
                                        embedding_status = 'indexed',
                                        indexed_at = CURRENT_TIMESTAMP
                                    WHERE chunk_id = %s
                                """, (embedding, settings.OLLAMA_EMBED_MODEL, chunk_id))
                    indexed_count += len(rows)
                    consecutive_failures = 0
                except memory.InsufficientMemoryError as exc:
                    VectorDB._mark_chunks_failed(chunk_ids)
                    failed_count += len(rows)
                    stop_reason = "out_of_memory"
                    logger.error("Indexing stopped: local model is out of memory")
                    raise memory.InsufficientMemoryError(
                        f"{exc} Indexing is resumable: {indexed_count} chunks "
                        "were committed. Retry POST /documents/{id}/index after "
                        "freeing memory."
                    ) from exc
                except Exception as e:
                    VectorDB._mark_chunks_failed(chunk_ids)
                    failed_count += len(rows)
                    consecutive_failures += 1
                    logger.error(f"Batch indexing failed: {type(e).__name__}")
                    stop_reason = type(e).__name__
                    if consecutive_failures >= max_failures:
                        logger.error(
                            f"Stopping indexing after {consecutive_failures} "
                            "consecutive failed batches"
                        )
                        break

            status_info = VectorDB.get_indexing_status(document_id)
            if status_info["failed"] == 0 and status_info["pending"] == 0:
                result_status = "complete"
            elif indexed_count > 0:
                result_status = "partial"
            else:
                result_status = "failed" if failed_count else "complete"

            return {
                "status": result_status,
                "indexed": indexed_count,
                "failed": failed_count,
                "processed": processed_count,
                "total": status_info["total"],
                "remaining": status_info["pending"] + status_info["failed"],
                "batch_size": batch_size,
                "resumable": True,
                "stop_reason": stop_reason,
            }
        finally:
            conn.close()

    @staticmethod
    def _mark_chunks_failed(chunk_ids: List) -> None:
        """Best-effort failure marker in its own committed transaction."""
        if not chunk_ids:
            return
        conn = VectorDB.get_connection()
        conn.autocommit = True
        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE document_chunks SET embedding_status = 'failed' "
                        "WHERE chunk_id = %s",
                        [(chunk_id,) for chunk_id in chunk_ids],
                    )
        except Exception as exc:
            logger.warning(f"Could not mark failed chunks: {type(exc).__name__}")
        finally:
            conn.close()
    
    @staticmethod
    async def search_similar(query: str, top_k: int = None, document_id: str = None) -> List[Dict]:
        """
        Search for similar chunks. Preserves all metadata.
        """
        if top_k is None:
            top_k = settings.TOP_K_CHUNKS
        top_k = clamp_int(top_k, 1, settings.MAX_TOP_K, settings.TOP_K_CHUNKS)
        
        try:
            query_embedding = await embedding_service.embed_text(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {type(e).__name__}")
            raise
        
        conn = VectorDB.get_connection()
        try:
            cur = conn.cursor()
            
            if document_id:
                cur.execute("""
                    SELECT chunk_id, document_id, page_number, chunk_index, 
                           chunk_text, char_count, 
                           1 - (embedding <=> %s::vector) as similarity
                    FROM document_chunks
                    WHERE embedding_status = 'indexed'
                    AND embedding IS NOT NULL
                    AND document_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, document_id, query_embedding, top_k))
            else:
                cur.execute("""
                    SELECT chunk_id, document_id, page_number, chunk_index, 
                           chunk_text, char_count,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM document_chunks
                    WHERE embedding_status = 'indexed'
                    AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, top_k))
            
            rows = cur.fetchall()
            cur.close()
            
            results = []
            for row in rows:
                results.append({
                    "chunk_id": str(row[0]),
                    "document_id": str(row[1]),
                    "page_number": row[2],
                    "chunk_index": row[3],
                    "text": row[4],
                    "char_count": row[5],
                    "similarity": float(row[6])
                })
            return results
        finally:
            conn.close()
    
    @staticmethod
    def get_indexing_status(document_id: str) -> Dict:
        """Get indexing status for a document."""
        conn = VectorDB.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE embedding_status = 'indexed') as indexed,
                    COUNT(*) FILTER (
                        WHERE embedding_status = 'pending'
                           OR embedding_status IS NULL
                    ) as pending,
                    COUNT(*) FILTER (WHERE embedding_status = 'failed') as failed
                FROM document_chunks
                WHERE document_id = %s
            """, (document_id,))
            row = cur.fetchone()
            cur.close()
            total = row[0] or 0
            indexed = row[1] or 0
            return {
                "total": total,
                "indexed": indexed,
                "pending": row[2] or 0,
                "failed": row[3] or 0,
                "percent_complete": round(indexed * 100.0 / total, 1) if total else 0.0,
            }
        finally:
            conn.close()
