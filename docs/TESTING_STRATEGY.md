# Testing Strategy

This project uses **pytest** for backend testing. Tests are designed to run locally without external paid services.

## Goals

- Validate modular boundaries (ingestion, extraction, graph, retrieval, generation)
- Ensure data integrity (IDs, chunks, citations, no duplicates)
- Verify local-only behavior (no network calls to paid APIs)
- Test resource-conscious configurations
- Maintain safety (no model-generated SQL/Cypher execution)

## Test Organization

```
backend/tests/
├── unit/           # Pure functions, schemas, utilities
├── integration/    # DB connections, API endpoints with test DBs
└── fixtures/       # Test data, mocks for Ollama
```

## Test Categories

### Unit Tests
- **Schemas** (`app/schemas/`) - Pydantic validation
- **Chunking** (`app/ingestion/chunker.py`) - Text splitting logic
- **Utils** - Configuration, helpers
- **Extraction parsing** - Entity/relation extraction from LLM output

### Integration Tests
- **Ingestion pipeline** - PDF → chunks (with test PDFs)
- **Vector store** - PostgreSQL + pgvector operations (test DB)
- **Graph store** - Neo4j operations (test DB or isolated)
- **Retrieval** - Hybrid search logic
- **Generation** - Answer with citations (mocked Ollama responses)
- **APIs** - FastAPI endpoints via TestClient

## Mocking Strategy

Since all inference must run locally via Ollama, tests use mocks for:
- `ollama` client calls (embeddings and completions)
- Heavy model operations
- External network calls

Use `unittest.mock` or `pytest-mock`.

## Running Tests

```bash
cd backend
pytest -v                    # All tests
pytest tests/unit -v         # Unit tests only
pytest tests/integration -v  # Integration tests only
pytest --cov=app --cov-report=html  # With coverage
```

## Test Requirements

- **No paid APIs** - All external calls must be mocked or use local test doubles
- **Isolation** - Tests should not depend on running Ollama/Neo4j/Postgres unless marked `@pytest.mark.integration`
- **Deterministic** - Use fixed fixtures, avoid randomness where possible
- **Fast unit tests** - Unit tests should run without DB/model dependencies
- **Idempotency** - Test retry safety for ingestion

## Markers

```python
@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.slow
```

## Coverage Targets (Initial)

- Schemas: 90%+
- Core logic (chunking, retrieval): 80%+
- APIs: 70%+
- Overall: 75%+

## CI Considerations

Tests should be runnable on Windows with local test databases or fully mocked. Integration tests requiring real services should be clearly marked and skippable.
