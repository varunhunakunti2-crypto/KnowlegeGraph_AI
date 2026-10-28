# TESTING — Mandatory Testing Rules

## Backend
- Use pytest for unit and integration tests.
- Test API validation, ingestion, chunking, extraction, retrieval, and citations.
- Test malformed files, empty results, database failures, and timeouts.

## Frontend
- Run linting and TypeScript checks.
- Test upload, document listing, chat, citations, and graph interactions.
- Verify loading, error, empty, and retry states.

## GraphRAG
- Test vector retrieval and graph retrieval independently.
- Test combined retrieval and source provenance.
- Test unsupported questions and insufficient evidence.
- Verify citations reference actual retrieved chunks.
- Test duplicate ingestion and safe deletion.

## Integration Tests
- Distinguish mocked tests from real Ollama, PostgreSQL, and Neo4j tests.
- Skip service-dependent tests only when explicitly configured to do so.
- Never describe mocked tests as real integration tests.

## Mandatory Reporting
1. Record the commands actually executed.
2. Report passed, failed, and skipped tests.
3. Never fabricate test results.
4. Never remove failing assertions merely to make tests pass.
5. Fix relevant failures before marking the task complete, or explain why they remain unresolved.