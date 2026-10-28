"""Document database operations."""
import psycopg
from app.core.config import settings
from typing import List, Dict, Optional
from datetime import datetime
import json


class DocumentDB:
    """PostgreSQL operations for documents."""
    
    @staticmethod
    def get_connection():
        return psycopg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )
    
    @staticmethod
    def init_table():
        """Initialize documents table."""
        conn = DocumentDB.get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id UUID PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size BIGINT NOT NULL,
                    page_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    error_message TEXT,
                    pages JSONB
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
                CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents(uploaded_at);
            """)
        conn.commit()
        conn.close()
    
    @staticmethod
    def create_document(doc_id: str, original_filename: str, stored_filename: str, 
                       file_path: str, file_size: int, page_count: int, status: str) -> None:
        conn = DocumentDB.get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (document_id, original_filename, stored_filename, 
                                     file_path, file_size, page_count, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (doc_id, original_filename, stored_filename, file_path, file_size, page_count, status))
        conn.commit()
        conn.close()
    
    @staticmethod
    def update_document_status(doc_id: str, status: str, pages: List[Dict] = None, 
                              error_message: str = None) -> None:
        conn = DocumentDB.get_connection()
        with conn.cursor() as cur:
            if pages is not None:
                cur.execute("""
                    UPDATE documents 
                    SET status = %s, processed_at = CURRENT_TIMESTAMP, pages = %s
                    WHERE document_id = %s
                """, (status, json.dumps(pages), doc_id))
            elif error_message is not None:
                cur.execute("""
                    UPDATE documents 
                    SET status = %s, processed_at = CURRENT_TIMESTAMP, error_message = %s
                    WHERE document_id = %s
                """, (status, error_message, doc_id))
            else:
                cur.execute("""
                    UPDATE documents 
                    SET status = %s, processed_at = CURRENT_TIMESTAMP
                    WHERE document_id = %s
                """, (status, doc_id))
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_document(doc_id: str) -> Optional[Dict]:
        conn = DocumentDB.get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT document_id, original_filename, file_size, page_count, 
                       status, uploaded_at, processed_at, error_message, pages
                FROM documents WHERE document_id = %s
            """, (doc_id,))
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "document_id": str(row[0]),
            "original_filename": row[1],
            "file_size": row[2],
            "page_count": row[3],
            "status": row[4],
            "uploaded_at": row[5],
            "processed_at": row[6],
            "error_message": row[7],
            "pages": row[8] if row[8] else []
        }
    
    @staticmethod
    def list_documents() -> List[Dict]:
        conn = DocumentDB.get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT document_id, original_filename, file_size, page_count, 
                       status, uploaded_at, processed_at, error_message
                FROM documents ORDER BY uploaded_at DESC
            """)
            rows = cur.fetchall()
        conn.close()
        return [{
            "document_id": str(row[0]),
            "original_filename": row[1],
            "file_size": row[2],
            "page_count": row[3],
            "status": row[4],
            "uploaded_at": row[5],
            "processed_at": row[6],
            "error_message": row[7]
        } for row in rows]
    
    @staticmethod
    def delete_document(doc_id: str) -> Optional[Dict]:
        conn = DocumentDB.get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT file_path FROM documents WHERE document_id = %s", (doc_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                return None
            file_path = row[0]
            cur.execute("DELETE FROM documents WHERE document_id = %s", (doc_id,))
        conn.commit()
        conn.close()
        return {"file_path": file_path}
