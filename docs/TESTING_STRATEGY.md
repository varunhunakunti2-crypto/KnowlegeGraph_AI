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
├── integration/    # DB connections, API endpoints with test DBs; optional live E2E
├── evaluation/     # Deterministic GraphRAG E2E, metrics, synthetic corpus
└── fixtures/       # Test data, mocks for Ollama
frontend/tests/e2e/ # Playwright display tests (mocked API)
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

### Evaluation Tests (`backend/tests/evaluation/`)
- **Full GraphRAG E2E** - synthetic PDF → validation → extraction → chunking →
  embeddings → entity/relation extraction → graph storage → vector + graph
  retrieval → evidence combination → cited answer → deletion cleanup
- **Metrics** - retrieval hit/recall/precision/MRR, citation validity,
  faithfulness proxy, graph triple precision/recall/F1
- **Comparison** - vector-only vs GraphRAG at the same `top_k`, including a
  connected-fact question that a tight vector window misses
- **Negative controls** - out-of-corpus abstention, document-scope leak
  prevention, fabricated-citation rejection
- **Latency/memory** - per-stage wall clock and `tracemalloc` peak

These tests replace only external boundaries (embedding model, vector store,
Neo4j, LLM). Production ingestion, retrieval, generation, citation, and deletion
code runs unchanged. See [EVALUATION.md](EVALUATION.md) for the corpus, metrics,
results, and limitations.

### Frontend Display Tests (`frontend/tests/e2e/`)
- Playwright tests with mocked `/api/v1/chat/` and `/api/v1/graph/` responses
- Verify rendered answer, `Sources:` citations, React Flow nodes/edges, and the
  node-details panel
- Verify empty and API-failure fallback states

### Live Tests (`backend/tests/integration/test_live_e2e.py`)
- Skipped unless `KG_LIVE=1`
- Require the real backend, PostgreSQL/pgvector, Neo4j, Ollama, and a running
  frontend
- Upload, index, real graph extraction, chat, graph read, frontend serve check,
  deletion, and cross-store cleanup
- A second test stores a controlled extraction through real Neo4j, reads it back
  through the graph API, and deletes it, so edge storage is covered even when
  qwen3:4b omits a relationship endpoint
- Slow on CPU-only machines; run manually before releases, never in CI

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
pytest -m evaluation -v      # Deterministic GraphRAG evaluation
pytest --cov=app --cov-report=html  # With coverage

# Deterministic evaluation report (JSON optional)
python -m scripts.evaluate
python -m scripts.evaluate --json eval-report.json

# Optional live stack test (must be enabled explicitly)
$env:KG_LIVE = "1"
python -m pytest tests/integration/test_live_e2e.py -v -s
```

Frontend:

```bash
cd frontend
npm run test:e2e
```

The Playwright config reuses a dev server on `http://localhost:3000` or starts
one. Install browsers once with `npx playwright install chromium-headless-shell`.

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
@pytest.mark.evaluation  # deterministic, no external services
@pytest.mark.live        # requires KG_LIVE=1 and the real stack
@pytest.mark.slow
```

## Coverage Targets (Initial)

- Schemas: 90%+
- Core logic (chunking, retrieval): 80%+
- APIs: 70%+
- Overall: 75%+

## CI Considerations

Tests should be runnable on Windows with local test databases or fully mocked. Integration tests requiring real services should be clearly marked and skippable.

- **CI (required):** `pytest -q` in `backend/` (includes deterministic
  `evaluation` tests) and `npm run test:e2e` in `frontend/`. Neither needs
  Ollama, PostgreSQL, Neo4j, or paid APIs.
- **Live (manual):** set `KG_LIVE=1` and run
  `tests/integration/test_live_e2e.py` against a running stack. It is slow and
  is skipped by default.
- **Failures:** the deterministic evaluation exits non-zero when a check fails;
  the live test fails (not skips) when enabled but the stack is unavailable.

See [EVALUATION.md](EVALUATION.md) for metrics, the synthetic corpus, the
vector-only vs GraphRAG comparison, and known limitations.
