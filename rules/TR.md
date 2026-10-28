# TR — Technical Rules

## Approved Stack
- Frontend: Next.js, React, TypeScript, Tailwind CSS
- Backend: Python 3.12, FastAPI, Pydantic
- Local AI: Ollama with qwen3:4b
- Embeddings: nomic-embed-text
- Graph database: Neo4j Community Edition
- Vector database: PostgreSQL with pgvector
- PDF processing: PyMuPDF
- Graph visualization: React Flow

## Architecture
PDF upload → validation → text extraction → chunking → embeddings and entity extraction → PostgreSQL/pgvector and Neo4j → hybrid retrieval → answer generation → citations.

## Mandatory Rules
1. Keep ingestion, extraction, graph, retrieval, and generation modules separate.
2. Use typed request and response schemas.
3. Use database migrations and appropriate indexes.
4. Verify embedding dimensions before defining vector columns.
5. Use consistent embedding models for indexing and querying.
6. Preserve document IDs, chunk IDs, page numbers, and source evidence.
7. Make ingestion safe to retry without creating duplicate records.
8. Limit graph traversal depth and retrieval result counts.
9. Never execute model-generated SQL or Cypher.
10. Keep model context length, concurrency, and batch sizes configurable.
11. All AI inference must run locally; no paid APIs or silent cloud fallbacks.
12. Avoid unnecessary microservices and excessive memory usage.

## Hardware Constraints
Target Windows, NVIDIA RTX 4050 with 6 GB VRAM, and 16 GB RAM.

Measure actual resource usage where possible. Do not claim GPU acceleration without verification.