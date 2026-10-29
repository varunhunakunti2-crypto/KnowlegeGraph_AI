# GraphRAG Evaluation

This document describes the end-to-end evaluation added for KnowledgeGraph AI, the
results of the deterministic run, and the limits of what those results prove. It
complements [TESTING_STRATEGY.md](TESTING_STRATEGY.md).

The evaluation has three layers:

1. **Deterministic suite (CI)** - production pipeline code, synthetic PDFs, test
   doubles only at the external boundaries (embeddings, vector DB, Neo4j, LLM).
2. **Frontend display suite (Playwright)** - real browser, mocked API responses.
3. **Optional live suite** - real backend, PostgreSQL/pgvector, Neo4j, Ollama and
   qwen3:4b / nomic-embed-text. Disabled unless `KG_LIVE=1`.

No private documents, paid APIs, or network model calls are used by the
deterministic or frontend suites.

## Reproduce

From the repository root:

```powershell
# Deterministic evaluation report (10 checks, JSON optional)
cd backend
.\.venv\Scripts\python.exe -m scripts.evaluate
.\.venv\Scripts\python.exe -m scripts.evaluate --json eval-report.json

# Deterministic evaluation tests only
.\.venv\Scripts\python.exe -m pytest -m evaluation -v

# Full backend suite (deterministic evaluation runs as part of it)
.\.venv\Scripts\python.exe -m pytest -q

# Frontend display tests (starts/reuses http://localhost:3000)
cd ..\frontend
npm run test:e2e

# Optional live stack test (requires backend :8000, frontend :3000, DBs, Ollama)
cd ..\backend
$env:KG_LIVE = "1"
.\.venv\Scripts\python.exe -m pytest tests/integration/test_live_e2e.py -v -s
```

The deterministic runner exits non-zero if any check fails. Use
`--no-strict` only when generating an exploratory report.

## Deterministic suite design

`backend/tests/evaluation/`:

| File | Purpose |
| --- | --- |
| `corpus.py` | Three synthetic documents (8 pages), gold entities/relations, deliberate noise, eight questions with expected sources |
| `pdfs.py` | Minimal deterministic PDF writer and byte builder |
| `doubles.py` | TF-IDF hashed embedder, in-memory vector store, in-memory graph store, scripted extraction/answer LLMs |
| `metrics.py` | Retrieval, citation, faithfulness, graph-triple, latency, and memory metrics |
| `pipeline.py` | Runs the real ingestion, extraction, retrieval, generation, citation, and deletion code |
| `test_graphrag_e2e.py` | 12-step end-to-end flow plus deletion cleanup |
| `test_evaluation_and_comparison.py` | Metric negative controls, vector-only vs GraphRAG, latency/memory |

The doubles sit only at these boundaries:

- **Embedding model** -> deterministic TF-IDF hashed vectors (same dimension as
  `EMBEDDING_DIMENSION`).
- **pgvector** -> in-memory vector store with the same search/delete contract.
- **Neo4j** -> in-memory graph store with the same document scoping, provenance,
  deduplication, and delete rules.
- **Ollama LLM** -> fixed JSON for entity/relation extraction and a scripted
  extractive answerer for generation.

Everything between those boundaries is production code: `PDFValidator`,
`PDFExtractor`, `TextChunker`, `GraphExtractionService`, `GraphRAGRetriever`,
`AnswerGenerator`, citation validation, and the cleanup accounting.

The scripted answerer only produces an answer when the retrieved evidence
contains the expected phrase, and it cites only real evidence chunk IDs. If
retrieval misses the fact, the run abstains and the failure is visible in the
metrics instead of being hidden by a fluent fake answer.

## Synthetic corpus and question set

Three PDFs, eight pages, eight chunks at the default 512-character chunk size:

| Document | Pages | Topics |
| --- | --- | --- |
| `ada_lovelace.pdf` | 3 | Babbage, Analytical Engine, Ada Lovelace, 1843 notes |
| `apollo11.pdf` | 3 | Apollo 11, NASA, Armstrong, Moon, trumpet |
| `photosynthesis.pdf` | 2 | Light, chlorophyll, chloroplasts, glucose |

| Question ID | Question | Expected answer | Expected sources |
| --- | --- | --- | --- |
| `q_babbage` | Who designed the Analytical Engine? | Charles Babbage | `ada_lovelace` p1 |
| `q_ada` | Who worked with Charles Babbage on the Analytical Engine? | Ada Lovelace | `ada_lovelace` p2 |
| `q_1843` | What did Ada Lovelace publish in 1843? | translation | `ada_lovelace` p2 |
| `q_apollo` | Who was the first person to walk on the Moon? | Neil Armstrong | `apollo11` p2 |
| `q_bridge` | Which instrument did the first person to walk on the Moon play? | trumpet | `apollo11` p3 (graph) |
| `q_multi` | Which organization launched the mission that Neil Armstrong flew on? | NASA | `apollo11` p1 + p2 (graph) |
| `q_missing` | What is the capital of France? | abstain | none (out of corpus) |
| `q_scope` | Who designed the Analytical Engine? (scoped to `photosynthesis`) | abstain | none; no Ada/Babbage leak |

`q_bridge` is the key GraphRAG case: the answer sentence shares almost no
vocabulary with the question, so a tight vector window misses it, while the
graph reaches it through the Neil Armstrong entity. `q_multi` requires joining
facts across two pages. `q_missing` and `q_scope` are negative controls.

## Metrics

| Metric | Definition | What it catches |
| --- | --- | --- |
| Retrieval hit@k / recall@k / precision@k / MRR | Page-level overlap between retrieved evidence and expected sources | Missed sources, ranking, noise |
| Citation validity | Every citation maps to a retrieved chunk with matching document and page | Fabricated or mismatched citations |
| Faithfulness | Fraction of cited sentences whose content tokens overlap the cited evidence (threshold 0.6); abstentions score 1.0 | Uncited claims, claims unsupported by their citation |
| Graph precision / recall / F1 | Normalized subject-predicate-object triples vs gold annotations | Dropped facts, hallucinated relations, noise handling |
| Latency | Wall-clock per stage (validation, extraction, chunking, indexing, graph extraction, retrieval, generation) | Regressions in pipeline wiring |
| Peak traced memory | `tracemalloc` peak during the deterministic run | Gross allocation regressions |

Retrieval recall is measured over all evidence the generator actually sees.
The graph channel is not bounded by `top_k`, so a GraphRAG run can have more
evidence than the vector window; the comparison therefore also reports the
vector window size and precision.

## Results snapshot (deterministic run)

Command: `backend/.venv/Scripts/python.exe -m scripts.evaluate` (top_k=2).
Snapshot from 2026-10-29 on Windows, Python 3.14, no live services.

```
10/10 checks passed
```

| Check | Result |
| --- | --- |
| Ingestion indexes every extracted chunk | PASS (3/3, 3/3, 2/2) |
| Graph extraction recovers every gold relationship | PASS (recall 1.00) |
| All citations point at retrieved evidence | PASS |
| GraphRAG answers connected-fact questions (`q_bridge`, `q_multi`) | PASS |
| GraphRAG answers single-hop questions | PASS |
| Out-of-corpus question abstains without citations | PASS |
| Document scope does not leak entities | PASS |
| Faithfulness of every non-abstained answer | PASS (1.00 for all six) |
| Deletion removes vector, graph, chunk, and file data | PASS (all counts 0, file removed) |
| Vector-only misses the connected fact at the same window | PASS (`q_bridge` recall 0.00) |

Ingestion and graph accuracy:

| Document | Pages | Chunks | Indexed | Entities | Relationships |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ada_lovelace` | 3 | 3 | 3 | 8 | 10 |
| `apollo11` | 3 | 3 | 3 | 11 | 8 |
| `photosynthesis` | 2 | 2 | 2 | 9 | 7 |

Graph triples against gold: **precision 0.88, recall 1.00, F1 0.94** (25
predicted, 22 gold). The three extra triples come from the deliberate noise
relationships: the sanitiser drops invalid entity types but normalizes invalid
predicates to `RELATED_TO` rather than discarding the relationship.

Per-question GraphRAG results:

| Question | Evidence | Recall | Precision | Answer OK | Citations | Faithfulness |
| --- | ---: | ---: | ---: | :---: | :---: | ---: |
| `q_babbage` | 20 | 1.00 | 0.12 | yes | 1/1 | 1.00 |
| `q_ada` | 20 | 1.00 | 0.12 | yes | 1/1 | 1.00 |
| `q_1843` | 20 | 1.00 | 0.12 | yes | 1/1 | 1.00 |
| `q_apollo` | 20 | 1.00 | 0.12 | yes | 1/1 | 1.00 |
| `q_bridge` | 9 | 1.00 | 0.33 | yes | 1/1 | 1.00 |
| `q_multi` | 9 | 1.00 | 0.67 | yes | 1/1 | 1.00 |
| `q_missing` | 20 | n/a | n/a | abstained | 0/0 | 1.00 |
| `q_scope` | 6 | n/a | n/a | abstained | 0/0 | 1.00 |

The low precision on unscoped questions is expected: graph retrieval is
document-scoped but question-independent, so it appends evidence for every
entity/relation in those documents. Recall is unaffected because the graph
channel is not limited by `top_k`; precision is the cost.

Latency and memory (deterministic doubles; **not** production timings):

| Stage | n | Total (s) | p50 (s) | p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| `validate_pdf` | 3 | 0.048 | 0.014 | 0.020 |
| `extract_pages` | 3 | 0.008 | 0.003 | 0.003 |
| `chunk` | 3 | 0.000 | 0.000 | 0.000 |
| `index_embeddings` | 8 | 0.009 | 0.001 | 0.001 |
| `extract_graph` | 8 | 0.006 | 0.001 | 0.001 |
| `retrieve` | 16 | 0.043 | 0.002 | 0.005 |
| `generate` | 16 | 0.004 | 0.000 | 0.001 |

Peak traced memory: **1.40 MB**. This covers Python allocations only; it
excludes native PyMuPDF allocations, database processes, and Ollama.

## Vector-only vs GraphRAG

Same corpus, same question set, same `top_k=2` vector window. GraphRAG recall is
measured over all returned evidence, including the unbounded graph channel.

| Question | Vector recall | GraphRAG recall | Vector precision | GraphRAG precision | Vector answers | GraphRAG answers |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| `q_babbage` | 1.00 | 1.00 | 0.50 | 0.12 | yes | yes |
| `q_ada` | 1.00 | 1.00 | 0.50 | 0.12 | yes | yes |
| `q_1843` | 1.00 | 1.00 | 0.50 | 0.12 | yes | yes |
| `q_apollo` | 1.00 | 1.00 | 0.50 | 0.12 | yes | yes |
| `q_bridge` | 0.00 | 1.00 | 0.00 | 0.33 | no (abstains) | yes |
| `q_multi` | 1.00 | 1.00 | 1.00 | 0.67 | yes | yes |

Interpretation:

- Vector-only is sufficient for single-hop questions when the answer shares
  vocabulary with the question.
- It fails `q_bridge` at a tight window because the answer sentence does not
  share the question's vocabulary.
- GraphRAG recovers that fact through the entity graph and never retrieves a
  subset of the vector evidence, at the cost of lower precision on unscoped
  questions.
- This is a property of the retrieval design on a small controlled corpus, not a
  universal quality claim about GraphRAG.

## Frontend display verification

`frontend/tests/e2e/chat-graph.spec.ts` uses Playwright with mocked API routes
(no backend or models required):

| Test | Assertion |
| --- | --- |
| Chat answer + citation | Rendered answer, `Sources:`, `ada_lovelace.pdf - Page 1` |
| Chat request body | Typed question is sent to `POST /api/v1/chat/` without a document ID |
| Graph render | React Flow nodes, `DESIGNED` edge label, node details (`Label`, `Type`, `ID`) |
| Empty graph | `No graph data available` |
| Graph API failure | Page falls back to the same empty state |

Run: `cd frontend; npm run test:e2e` (config starts `npm run dev` or reuses an
existing server on port 3000). Latest run: **5 passed**.

These tests verify the display contract only. They mock the API, so they do not
prove backend data shape beyond the contract test in
`backend/tests/evaluation/test_api_contract.py`, and they do not exercise real
upload, retrieval, or generation.

## Optional live suite

`backend/tests/integration/test_live_e2e.py` is skipped unless `KG_LIVE=1`. When
enabled it fails (rather than silently skipping) if the stack is unavailable. It
uploads a one-page synthetic PDF, extracts pages, indexes embeddings, runs real
qwen3:4b entity/relation extraction, asks a real question, reads the graph and
entity details, checks that the frontend is served, deletes the document, and
verifies PostgreSQL, Neo4j, and upload-directory cleanup.

Environment variables: `KG_BASE_URL` (default `http://127.0.0.1:8000`),
`KG_FRONTEND_URL` (default `http://127.0.0.1:3000`), `KG_LIVE=1`.

The module contains two tests:

1. **Real model flow.** Upload, page extraction, chunking, nomic embeddings,
   qwen3:4b graph extraction, chat, entity search/detail, frontend serve check,
   delete, and cross-store cleanup. Assertions are structural: pages/chunks
   exist, at least one entity is extracted, citations belong to the test
   document/page, graph nodes exist, and the answer contains the fixture
   subject (`Mara Voss`/`Mara`). If the model reports a usable relationship, the
   graph must expose an edge; if it reports none, the test records that instead
   of pretending the edge exists.
2. **Controlled Neo4j round trip.** Builds a fixed extraction result (3
   entities, 2 relationships), stores it through the real `GraphDB`, reads it
   back through both `GraphDB` and `GET /api/v1/graph/`, and verifies deletion.
   This covers real Neo4j relationship storage and the graph API independently
   of model variance.

Run result (2026-10-29, RTX 4050 laptop, CPU-only Ollama, backend on
`127.0.0.1:8010` via `KG_BASE_URL`; the default is `127.0.0.1:8000`):
**2 passed in 202.51s**.

| Observation | Result |
| --- | --- |
| Model extraction | 1 entity, 0 usable relationships |
| Graph API | 1 node, 0 edges |
| Chat | Answered with a citation to the test page |
| Ready / upload / index | 7.76 s / 15.13 s / 17.77 s |
| Graph extraction / chat / delete | 30.23 s / 95.12 s / 10.10 s |
| Backend RSS (optional sample) | ~119.7 MB |

The zero-edge result is a model-quality observation, not a graph-wiring
failure: qwen3:4b extracted only one entity and omitted the other endpoint, so
there was no edge to store. The controlled round-trip test is what guarantees
the real Neo4j edge path. On CPU-only machines qwen3:4b extraction and answer
generation can take minutes per chunk; latency and RSS are printed as
diagnostics, not enforced thresholds.

## Known limitations

1. **No similarity threshold.** Out-of-corpus questions still return low-scoring
   evidence. Correct behavior depends on the generator abstaining; the scripted
   answerer does, but real model behavior is unmeasured here.
2. **Question-independent graph retrieval.** The graph channel returns all
   evidence for the scoped document(s), not just question-relevant entities.
   This raises recall and lowers precision, as the `0.12` precision column shows.
3. **Graph evidence is not bounded by `top_k`.** Comparisons measure recall over
   everything the generator sees and report precision/window size alongside it.
4. **Faithfulness is a lexical proxy.** It is content-token overlap, not
   entailment. Verbatim extraction scores 1.0; a valid paraphrase may score low,
   and an entity substitution can pass if enough surrounding words overlap
   (for example "Napoleon designed the Analytical Engine" against Babbage text).
   The suite's negative control uses a wholly unrelated claim, which fails
   reliably.
5. **Deterministic doubles are not models.** The TF-IDF embedder is not
   nomic-embed-text, and the scripted answerer is not qwen3:4b. The deterministic
   suite tests pipeline wiring, retrieval logic, citation validation, metrics,
   and cleanup; it does not estimate model quality.
6. **Small synthetic corpus.** Eight pages and eight questions are enough for
   deterministic regression tests and a clear `q_bridge` contrast, not for
   broad generalization claims.
7. **Latency and memory are in-process.** They exclude databases, native
   libraries, and model inference, so they are not production SLOs.
8. **Frontend tests mock the API.** They prove the UI renders the documented
   response shape, not that a real upload/chat/graph round trip works in the
   browser.
9. **Live suite is narrow and model-dependent.** It checks structural
   correctness and one semantic keyword, not full answer quality. The real
   qwen3:4b model may extract a relationship whose endpoint it never listed as
   an entity, leaving the graph with nodes but no edge; the live test records
   this and the controlled Neo4j round-trip test covers edge storage separately.
   The live suite is intentionally excluded from CI.

## CI guidance

- Run `pytest -q` (includes `-m evaluation`) for every change. It needs no
  services.
- Run `npm run test:e2e` after frontend changes; keep Playwright browsers cached
  in CI.
- Do not enable `KG_LIVE` in CI. Run it manually before releases or after
  changing ingestion, extraction prompts, retrieval, or generation.
- Treat metric changes as regressions only when the deterministic checks fail;
  otherwise inspect the JSON report to see whether recall, precision, citations,
  or faithfulness moved.
