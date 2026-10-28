"""Vector operations with pgvector."""
import psycopg
from app.core.config import settings
from app.services.embeddings import embedding_service
from app.core.logging import logger
from app.core.security import clamp_int
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
        """
        Index all chunks for a document. Idempotent - skips already indexed.
        """
        conn = VectorDB.get_connection()
        try:
            with conn.transaction():
                cur = conn.cursor()
                
                # Get unindexed chunks
                cur.execute("""
                    SELECT chunk_id, chunk_text 
                    FROM document_chunks 
                    WHERE document_id = %s 
                    AND (embedding_status IS NULL OR embedding_status != 'indexed')
                    ORDER BY chunk_index
                """, (document_id,))
                chunks = cur.fetchall()
                
                if not chunks:
                    return {"status": "complete", "indexed": 0, "message": "No chunks to index"}
                
                # Generate embeddings in batches
                indexed_count = 0
                batch_size = clamp_int(
                    settings.EMBEDDING_BATCH_SIZE, 1, settings.BATCH_SIZE, 16
                )
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    texts = [chunk[1] for chunk in batch]
                    chunk_ids = [chunk[0] for chunk in batch]
                    
                    try:
                        embeddings = await embedding_service.embed_batch(texts)
                        
                        for chunk_id, embedding in zip(chunk_ids, embeddings):
                            cur.execute("""
                                UPDATE document_chunks 
                                SET embedding = %s, 
                                    embedding_model = %s,
                                    embedding_status = 'indexed',
                                    indexed_at = CURRENT_TIMESTAMP
                                WHERE chunk_id = %s
                            """, (embedding, settings.OLLAMA_EMBED_MODEL, chunk_id))
                            indexed_count += 1
                    except Exception as e:
                        # Mark as failed
                        for chunk_id in chunk_ids:
                            cur.execute("""
                                UPDATE document_chunks 
                                SET embedding_status = 'failed'
                                WHERE chunk_id = %s
                            """, (chunk_id,))
                        logger.error(f"Batch indexing failed: {type(e).__name__}")
                
                cur.close()
            conn.commit()
            return {"status": "complete", "indexed": indexed_count, "total": len(chunks)}
        except Exception as e:
            conn.rollback()
            raise
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
                    COUNT(*) FILTER (WHERE embedding_status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE embedding_status = 'failed') as failed
                FROM document_chunks
                WHERE document_id = %s
            """, (document_id,))
            row = cur.fetchone()
            cur.close()
            return {
                "total": row[0],
                "indexed": row[1],
                "pending": row[2],
                "failed": row[3]
            }
        finally:
            conn.close()
