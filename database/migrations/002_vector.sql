-- Migration 002: Add vector embeddings
-- Verify embedding dimensions before creating vector column

-- Add embedding column to document_chunks
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector(768);
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'nomic-embed-text';
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_status TEXT DEFAULT 'pending' CHECK (embedding_status IN ('pending', 'indexed', 'failed'));

-- Create index for vector similarity search
-- Using IVFFlat for smaller datasets, can be changed to HNSW for larger
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Index for filtering
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_status ON document_chunks(embedding_status);
CREATE INDEX IF NOT EXISTS idx_chunks_indexed_at ON document_chunks(indexed_at);
