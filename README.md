# KnowledgeGraph AI

KnowledgeGraph AI is a fully local GraphRAG document assistant for Windows. It
uploads PDFs, extracts text page by page, chunks and embeds the text, extracts
entities and relationships into a Neo4j graph, retrieves evidence with both
vector search and bounded graph traversal, and generates answers with
document/chunk/page citations. All inference goes through a local Ollama
instance; there are no paid APIs and no silent cloud fallbacks.

This repository is a working local development project, not a production
service. It has no authentication or multi-user isolation and is intended to run
on a single trusted machine with the API and databases bound to loopback.

## What is implemented

- PDF upload with MIME/signature/page-size limits, active-content checks, atomic
  file writes, and path-traversal protection.
- Page-preserving text extraction with PyMuPDF.
- Configurable chunking with overlap and per-document limits.
- Idempotent chunk persistence in PostgreSQL.
- Manual vector indexing with `nomic-embed-text` and pgvector.
- Structured entity/relationship extraction with `qwen3:4b`, validated against
  fixed entity-type and predicate vocabularies before it reaches Neo4j.
- Neo4j storage of extracted entities, relationships, and source evidence.
- Graph API for the full graph, document-scoped graphs, entity search, entity
  details, neighbors, and edge evidence.
- Hybrid retrieval (vector similarity plus graph evidence), document-scoped
  retrieval, evidence deduplication, context budgeting, answer generation, and
  citation validation.
- Consistent deletion across the uploaded file, PostgreSQL rows, and Neo4j
  evidence (graph deletion is best-effort and reported in the response).
- Next.js UI for documents, chat with citations, graph visualization, and
  service/model status.
- Deterministic backend tests and evaluation, optional live tests, and
  Playwright display tests.

The browser UI currently exposes upload/list/delete, chat, and graph viewing.
Indexing and graph extraction are deliberately separate API steps and **do not
have buttons in the UI**. See [First document workflow](#first-document-workflow).

## Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js 16.4.3 (App Router), React 19.3, TypeScript 5.9, Tailwind CSS 4.3, React Flow 11.11 |
| Backend | Python 3.12+ (tested on 3.14.5), FastAPI 0.162.0, Pydantic 2.14 |
| Local LLM | Ollama with `qwen3:4b` |
| Embeddings | Ollama with `nomic-embed-text` (768 dimensions) |
| Vector store | PostgreSQL 17 + pgvector 0.8.7 |
| Graph store | Neo4j 5.25 Community |
| PDF processing | PyMuPDF 2.0 |
| Backend tests | pytest 9.2, pytest-asyncio, deterministic evaluation suite |
| Frontend tests | Playwright 1.64 with mocked API responses |

## Architecture

```text
PDF
 |
 v
Upload endpoint
  - validate extension/MIME/signature/size/pages
  - reject active PDF content by default
  - write a server-named file atomically
  - persist document + page snapshots
  - extract text (PyMuPDF) -> chunk -> store chunks
 |
 +---------------------------- separate requests ----------------------------+
 |                                                                          |
 v                                                                          v
POST /documents/{id}/index                              POST /graph/extract/{id}
  - nomic-embed-text embeddings                           - qwen3:4b structured JSON
  - document_chunks.embedding (vector 768)                 - validate entity/predicate vocabularies
  - embedding_status = indexed                             - Neo4j Entity/RELATION/SOURCED_FROM
 |                                                                          |
 +-----------------------------------+--------------------------------------+
                                     |
                                     v
                          Hybrid retrieval (chat)
                           - vector search (pgvector)
                           - graph traversal (Neo4j, bounded)
                           - deduplicate + budget context
                                     |
                                     v
                          qwen3:4b answer generation
                           - answer only from evidence
                           - cite [chunk:...]
                           - abstain when evidence is missing
                                     |
                                     v
                     FastAPI response -> Next.js chat/graph UI
```

Main backend packages:

- `backend/app/ingestion/` - PDF validation, text extraction, chunking.
- `backend/app/services/` - Ollama LLM, embeddings, graph extraction.
- `backend/app/db/` - PostgreSQL, pgvector, Neo4j, chunk/document persistence.
- `backend/app/retrieval/` - hybrid GraphRAG retrieval.
- `backend/app/generation/` - grounded answer generation and citations.
- `backend/app/api/v1/` - REST endpoints.
- `backend/app/schemas/` - request/response and graph schemas.

## Project structure

```text
KnowledgeGraph AI/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # health, models, documents, chat, graph routes
│   │   ├── core/            # settings, logging, security, resource limits
│   │   ├── db/              # PostgreSQL, pgvector, Neo4j, chunks, documents
│   │   ├── generation/      # evidence-grounded answers and citations
│   │   ├── ingestion/       # validation, extraction, chunking
│   │   ├── retrieval/       # vector + graph hybrid retrieval
│   │   ├── schemas/         # Pydantic API and graph schemas
│   │   ├── services/        # Ollama LLM/embeddings, graph extraction
│   │   └── main.py          # FastAPI app and lifespan
│   ├── scripts/             # smoke, evaluation, benchmark, graph cleanup CLIs
│   ├── tests/
│   │   ├── unit/            # fast isolated tests
│   │   ├── integration/     # optional live E2E tests
│   │   └── evaluation/      # deterministic GraphRAG evaluation
│   ├── requirements.txt
│   └── pytest.ini
├── database/migrations/     # PostgreSQL schema and pgvector migrations
├── docs/                    # performance, test strategy, and evaluation notes
├── frontend/
│   ├── app/                 # App Router pages
│   ├── components/          # layout components
│   ├── lib/                 # browser API client
│   ├── tests/e2e/           # Playwright display tests
│   └── types/               # frontend API types
├── .env.example
├── docker-compose.dev.yml   # PostgreSQL/pgvector + Neo4j for local development
└── README.md
```

The empty `backend/app/extraction/`, `backend/app/graph/`, and
`backend/app/utils/` packages are legacy placeholders; the active implementation
lives in `services/`, `db/`, and `core/`.

## Prerequisites

- Windows 10/11 with PowerShell.
- Python 3.12 or newer. The audit environment used Python 3.14.5.
- Node.js 20.9 or newer and npm. The audit environment used Node 24.14.0 and
  npm 11.9.0.
- [Ollama for Windows](https://ollama.com/download). Keep it running locally.
- Docker Desktop if you use the bundled PostgreSQL and Neo4j services.
  Alternatively, provide your own PostgreSQL with pgvector and Neo4j 5.x.
- Roughly 4-5 GB free disk for Ollama models plus database storage.
- Target hardware: RTX 4050 6 GB VRAM, 16 GB RAM. For this machine, use the
  default profile for the best evidence budget, or `MEMORY_PROFILE=low-memory`
  when the browser, IDE, and other applications are also using memory. See
  [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for measured throughput and VRAM
  use; GPU acceleration is verified with `ollama ps`, not assumed.

## 1. Configure environment variables

From the repository root in PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace `POSTGRES_PASSWORD` and `NEO4J_PASSWORD` with local values. Keep the
matching values in `DATABASE_URL` if you use that field for external tooling.
The backend itself uses the separate `POSTGRES_*` settings.

Important:

- `.env` is gitignored. Never commit real credentials.
- The backend loads the repository-root `.env` automatically, regardless of the
  process working directory.
- With the bundled Docker Compose file, leave `NEO4J_USER=neo4j`; Compose creates
  the database user as `neo4j`. The Neo4j password is only applied when the
  volume is first initialized.
- `NEXT_PUBLIC_API_URL` is a browser-visible setting; it is not a secret.
- CORS origins must be explicit, comma-separated `http://` or `https://`
  origins. Wildcards are rejected.

### Low-memory configuration

The default profile is tuned for a machine with 16 GB RAM and a 6 GB GPU. If the
browser, IDE, and other applications make memory tight, set this in `.env`:

```dotenv
MEMORY_PROFILE=low-memory
```

The profile applies smaller defaults only to settings you have not set
explicitly. A fresh `.env` copied from `.env.example` contains most of those
settings explicitly, so comment out or delete the profile-sensitive lines if you
want the profile to control them. Values you intentionally leave in `.env`
always win. See [.env.low-memory.example](.env.low-memory.example) for the full
list and the lines to comment out. The low-memory profile:

- reduces the model context from 4096 to 2048 and the output limit from 2048 to
  384;
- uses a smaller prompt batch (128) and one concurrent model request;
- keeps only one of `qwen3:4b`/`nomic-embed-text` resident at a time;
- indexes in batches of 4 and stops after 2 consecutive failed batches;
- reduces retrieval breadth and graph extraction work.

Explicit values in `.env` always override the profile. The measured tradeoffs
for both profiles are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## 2. Start PostgreSQL and Neo4j

Start Docker Desktop, then from the repository root:

```powershell
docker compose -f docker-compose.dev.yml up -d
docker compose -f docker-compose.dev.yml ps
```

Both services should become `healthy`. The Compose file:

- runs `pgvector/pgvector:pg17` as `kg-postgres`;
- runs `neo4j:5.25-community` as `kg-neo4j`;
- binds PostgreSQL `5432`, Neo4j Bolt `7687`, and Neo4j Browser `7474` to
  `127.0.0.1` only;
- stores data in the named volumes `knowledgegraphai_kg_pg_data`,
  `knowledgegraphai_kg_neo4j_data`, and `knowledgegraphai_kg_neo4j_logs`;
- mounts `database/migrations/` into PostgreSQL's
  `/docker-entrypoint-initdb.d/`.

### Database initialization caveat

PostgreSQL runs the files in `/docker-entrypoint-initdb.d/` **only when the data
volume is empty**. On a new clone with a new volume, migrations 001, 002, and 003
are applied automatically. Changing migration files later does not re-run them
on an existing volume. The FastAPI startup only ensures that the `documents`
table exists; it does not apply the remaining schema or pgvector migration.

Check the schema:

```powershell
docker exec kg-postgres psql -U postgres -d knowledgegraph -c "\dt"
docker exec kg-postgres psql -U postgres -d knowledgegraph -c "SELECT extname, extversion FROM pg_extension;"
```

Expected tables: `documents`, `document_pages`, `document_chunks`,
`ingestion_jobs`. Expected extension: `vector`, version 0.8.x with the bundled
image.

If you have an existing PostgreSQL volume without the full schema, apply the
idempotent migrations manually:

```powershell
docker exec -i kg-postgres psql -U postgres -d knowledgegraph -f /docker-entrypoint-initdb.d/001_init.sql
docker exec -i kg-postgres psql -U postgres -d knowledgegraph -f /docker-entrypoint-initdb.d/002_vector.sql
docker exec -i kg-postgres psql -U postgres -d knowledgegraph -f /docker-entrypoint-initdb.d/003_document_pages.sql
```

For an external PostgreSQL server, make sure pgvector is installed first, create
the database, then run the same SQL files with `psql`. The application does not
create the vector extension for you.

To reset the bundled databases and re-run initialization, first stop the
services and **delete the volumes intentionally**. This destroys stored
documents, chunks, embeddings, users, and graph data:

```powershell
docker compose -f docker-compose.dev.yml down -v
docker compose -f docker-compose.dev.yml up -d
```

### Data model

PostgreSQL tables (created by migrations):

- `documents` - document ID, filenames, path, size, page count, status,
  timestamps, metadata, and the extracted `pages` JSONB snapshot (migration 003).
- `document_pages` - one row per page with extracted text and character count.
- `document_chunks` - chunk text, page/chunk indexes, optional pgvector
  `embedding vector(768)`, embedding model/status/timestamp.
- `ingestion_jobs` - schema defined for future background ingestion jobs; the
  current synchronous upload flow does not insert rows into it.

Neo4j graph schema:

```text
(:Document {id})
(:Entity {id, name, type, aliases, confidence})
(:Entity)-[:SOURCED_FROM {document_id, chunk_id, page_number, evidence}]->(:Document)
(:Entity)-[:RELATION {type, document_id, chunk_id, page_number, evidence, confidence}]->(:Entity)
```

Entity types are limited to `Person`, `Organization`, `Technology`, `Concept`,
`Location`, and `Event`. Predicates are limited to `WORKS_FOR`, `CREATED`,
`USES`, `LOCATED_IN`, `PART_OF`, `RELATED_TO`, and `MENTIONS`. Model output is
validated before storage. Graph visualization queries traverse relationships in
the outgoing direction so each relationship is returned once; retrieval uses a
bounded traversal to find related evidence.

## 3. Install and start Ollama models

Check that Ollama is running:

```powershell
ollama list
```

If the command cannot connect, start the Ollama app or run `ollama serve` in a
separate terminal. Then pull the two required models:

```powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text
ollama list
```

Expected model names are exactly `qwen3:4b` and `nomic-embed-text`. Approximate
download sizes in the audit environment were 2.5 GB and 274 MB. `qwen3:4b` is
used for generation and graph extraction; `nomic-embed-text` is used for both
document and query embeddings.

If you use a different embedding model, update `OLLAMA_EMBED_MODEL` and
`EMBEDDING_DIMENSION`, and plan a database migration and re-index. The vector
column is fixed at 768 dimensions in migration 002.

### Verify whether Ollama is using the GPU

Do not infer GPU use from the presence of an NVIDIA card. After the first model
call, check:

```powershell
ollama ps
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/status
```

`ollama ps` reports `100% GPU`, `100% CPU`, or a split. If it says `100% CPU`,
see [If Ollama reports 100% CPU](#if-ollama-reports-100-cpu). The backend's
`/api/v1/system/status` endpoint returns the measured state rather than assuming
CUDA is active.

## 4. Install the backend

From the repository root:

```powershell
Set-Location backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the venv's `python.exe` directly avoids PowerShell execution-policy issues
with `Activate.ps1`. The pinned versions in `backend/requirements.txt` are the
versions exercised by the passing test suite on Python 3.14.5. They require
Python 3.11 or newer because of current PyMuPDF and pydantic-settings releases.

If a package fails to build, confirm that you are using a 64-bit Python 3.12+
interpreter and not the Microsoft Store alias. You do not need a CUDA toolkit for
Ollama; install and verify Ollama separately.

## 5. Install the frontend

In a second PowerShell window from the repository root:

```powershell
Set-Location frontend
npm install
npx playwright install chromium
```

The frontend reads `NEXT_PUBLIC_API_URL` at build/dev-server start. If you change
it, restart the dev server. `frontend/lib/api.ts` is part of the repository; an
older `.gitignore` rule previously hid it, which is fixed.

## 6. Start the application

Keep Docker, Ollama, the backend, and the frontend running. Use separate
terminals.

### Terminal 1 - backend

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend is available at <http://127.0.0.1:8000>. Interactive API docs are at
<http://127.0.0.1:8000/docs>.

Verify readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health/ready
Invoke-RestMethod http://127.0.0.1:8000/api/v1/models/status
```

`/health/ready` returns HTTP 200 with a top-level `ready` boolean. Check the
boolean, not just the status code. `models/status` should report both models as
available.

### Terminal 2 - frontend

```powershell
Set-Location frontend
npm run dev
```

Open <http://localhost:3000>. The audit environment served the frontend from
port 3000.

### Optional local production build

The build was verified with:

```powershell
Set-Location frontend
npm run build
```

To serve that build locally, use `npm run start`. This is still a local,
single-user application; see the limitations section.

## First document workflow

The upload endpoint validates and chunks a PDF, but it does **not** create
embeddings or graph nodes. After upload:

1. Find the document ID from the Documents page list or the API.
2. Call the indexing endpoint for vector retrieval.
3. Optionally call graph extraction, then open Graph Explorer.

The Documents page does not currently provide Index or Extract graph buttons.
Use Swagger at `/docs` or the PowerShell calls below.

### Upload with PowerShell

Use `curl.exe` rather than the `curl` alias so that the `-F` option works in
Windows PowerShell as well as PowerShell 7:

```powershell
curl.exe -X POST -F "file=@C:\path\to\document.pdf;type=application/pdf" http://127.0.0.1:8000/api/v1/documents/upload
```

The response contains `document_id`, `filename`, `status`, and `message`. The
upload reaches `completed` after validation, text extraction, chunking, and
chunk persistence; it is not yet indexed for vector search.

### Index, extract, chat, graph

```powershell
$documents = Invoke-RestMethod http://127.0.0.1:8000/api/v1/documents/
$id = $documents[0].document_id

# Required for vector retrieval.
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/v1/documents/$id/index"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/documents/$id/index/status"

# Optional: build the Neo4j graph. This can take minutes on CPU-only Ollama.
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/v1/graph/extract/$id"

$body = @{
    question = "Who designed the Analytical Engine?"
    document_id = $id
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/chat/" `
    -ContentType "application/json" `
    -Body $body

Invoke-RestMethod "http://127.0.0.1:8000/api/v1/graph/?document_id=$id"
```

Indexing is idempotent: already indexed chunks are skipped. It is also
resumable: chunks are embedded and committed in small batches
(`INDEX_BATCH_SIZE`), so a crash or out-of-memory stop keeps completed work. A
failed batch is marked `failed` and retried by the next index request. If the
local model runs out of memory, the endpoint returns HTTP 503 with the free
RAM/VRAM and leaves the committed batches intact. Check progress with
`GET /api/v1/documents/{id}/index/status`.

Graph extraction is also safe to retry; entities and relationships are merged in
Neo4j, and evidence is updated. A document-scoped chat checks that the document
exists and restricts both retrieval channels to that document.

## API endpoints

All routes are under `/api/v1` except the root and generated docs.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic app name/version and `local_only` flag. |
| `GET` | `/api/v1/health` | Basic health response. |
| `GET` | `/api/v1/health/ready` | Probes Ollama models, PostgreSQL/pgvector, and Neo4j; check `ready`. |
| `GET` | `/api/v1/models/status` | Reports LLM and embedding model availability. |
| `GET` | `/api/v1/system/status` | Measured RAM/VRAM, resident models, pressure, and active limits. |
| `GET` | `/api/v1/system/gpu` | Fresh `nvidia-smi` sample, or `measured: false` when unavailable. |
| `POST` | `/api/v1/system/unload` | Unload this application's Ollama models immediately. |
| `POST` | `/api/v1/documents/upload` | Multipart PDF upload; validates, extracts, chunks, and stores. |
| `GET` | `/api/v1/documents/` | List document metadata. |
| `GET` | `/api/v1/documents/{document_id}` | Document metadata plus page snapshots. |
| `POST` | `/api/v1/documents/{document_id}/index` | Embed unindexed chunks with pgvector. |
| `GET` | `/api/v1/documents/{document_id}/index/status` | Total/indexed/pending/failed chunk counts. |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete file, chunks, embeddings, and graph evidence. |
| `POST` | `/api/v1/chat/` | Hybrid retrieval plus cited answer generation. |
| `GET` | `/api/v1/graph/` | Full graph, or document-scoped graph with `?document_id=...`. |
| `GET` | `/api/v1/graph/entities/search` | Entity name search with optional type/document filters. |
| `GET` | `/api/v1/graph/entities/{entity_id}` | Entity details, evidence, and neighbors. |
| `GET` | `/api/v1/graph/entities/{entity_id}/neighbors` | Bounded neighbor traversal with filters. |
| `GET` | `/api/v1/graph/edges/evidence` | Evidence for a source/target edge. |
| `POST` | `/api/v1/graph/extract/{document_id}` | Extract and store entities/relationships. |
| `DELETE` | `/api/v1/graph/document/{document_id}` | Delete graph evidence for a document. |

### Chat request/response

```json
{
  "question": "Who designed the Analytical Engine?",
  "document_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

`document_id` is optional; omit it to search all indexed documents. The response
contains `answer`, `citations`, `evidence`, `has_evidence`, and
`retrieval_metadata`. A citation includes `chunk_id`, `document_id`,
`document_name`, `page_number`, and `snippet`. When evidence is missing, the API
returns `has_evidence: false`, no citations, and an explicit insufficient-
evidence message instead of asking the model to answer from prior knowledge.

### Status and error codes

| Code | Meaning |
| --- | --- |
| `400` | Invalid/unsafe identifier, missing document ID, bad extension/MIME, empty/corrupt PDF, no chunks to extract, or invalid request data. |
| `404` | Document/entity not found. |
| `413` | Upload exceeds `MAX_UPLOAD_SIZE_MB`. |
| `422` | FastAPI/Pydantic request validation failure (for example, an empty or missing question). |
| `500` | Unexpected database/model failure. Error bodies are generic; stack traces, paths, SQL, and secrets are not returned. |
| `503` | Local model memory is insufficient (RAM/VRAM). The message includes measured headroom and suggests `MEMORY_PROFILE=low-memory`, smaller context/output limits, or a smaller model. |
| `504` | Upload processing or graph extraction timed out. |

The application uses parameterized SQL/Cypher values, server-generated file
names, path containment checks, explicit CORS origins, and generic error
messages. Uploaded PDFs with active content (`/JavaScript`, `/Launch`,
`/EmbeddedFile`, `/OpenAction`) are rejected unless
`ALLOW_PDF_ACTIVE_CONTENT=true` is explicitly set for a trusted corpus.

## Environment variable reference

The full working set is in `.env.example`. Defaults below come from
`backend/app/core/config.py`.

### Core and Ollama

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENV` | `development` | Environment label. |
| `DEBUG` | `true` | Debug flag. |
| `BACKEND_HOST` | `127.0.0.1` | Used by app startup warning and `python -m app.main`; uvicorn flags bind the actual server. |
| `BACKEND_PORT` | `8000` | Used by `python -m app.main`; uvicorn flags set the actual port. |
| `BACKEND_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Exact allowed browser origins; no wildcards. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Browser-visible backend base URL. |
| `LOG_LEVEL` | `INFO` | Python log level. |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama endpoint. |
| `OLLAMA_MODEL` | `qwen3:4b` | Generation and graph-extraction model. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model. |
| `OLLAMA_TIMEOUT` | `300` | Ollama client timeout in seconds. |
| `OLLAMA_TEMPERATURE` | `0.1` | Generation temperature. |
| `OLLAMA_MAX_TOKENS` | `2048` | Generation token limit; low-memory profile uses `384`. |
| `OLLAMA_MAX_RETRIES` | `2` | Maximum generation attempts; `0` is treated as one attempt. |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long Ollama keeps a model after a request (`30s`, `5m`, `-1`, etc.). |
| `OLLAMA_IDLE_UNLOAD_SECONDS` | `900` | Unload this app's models after idle time; `0` disables. |
| `OLLAMA_NUM_BATCH` | `512` | Prompt-processing batch size; lower values reduce peak VRAM. |
| `OLLAMA_SINGLE_MODEL_RESIDENT` | `false` | Keep only one app model (LLM or embeddings) resident at a time. |

### Databases

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt endpoint. |
| `NEO4J_USER` | `neo4j` | Neo4j user; keep `neo4j` with bundled Compose. |
| `NEO4J_PASSWORD` | `changeme` | Neo4j password; replace locally. |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name. |
| `NEO4J_QUERY_TIMEOUT_SECONDS` | `10` | Validated setting; not currently passed to Neo4j driver calls. |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host. |
| `POSTGRES_PORT` | `5432` | PostgreSQL port. |
| `POSTGRES_DB` | `knowledgegraph` | Database name. |
| `POSTGRES_USER` | `postgres` | Database user. |
| `POSTGRES_PASSWORD` | `changeme` | Database password; replace locally. |
| `DATABASE_URL` | `postgresql+psycopg://postgres:changeme@localhost:5432/knowledgegraph` | Informational for external tooling; the app uses `POSTGRES_*`. |
| `DB_CONNECT_TIMEOUT_SECONDS` | `5` | Database connection timeout. |

### Ingestion, retrieval, and limits

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_DIMENSION` | `768` | Must match the `vector(768)` column and embedding model. |
| `EMBEDDING_BATCH_SIZE` | `32` | Requested embedding batch size, additionally capped by `BATCH_SIZE`. |
| `CHUNK_SIZE` | `512` | Maximum chunk window in characters. |
| `CHUNK_OVERLAP` | `50` | Chunk overlap; bounded below chunk size. |
| `TOP_K_CHUNKS` | `10` | Default vector result count. |
| `TOP_K_ENTITIES` | `10` | Configured graph/entity target; not separately enforced by the current retrieval query. |
| `GRAPH_TRAVERSAL_DEPTH` | `2` | Bounded graph traversal depth (capped at `MAX_GRAPH_TRAVERSAL_DEPTH`). |
| `MAX_CONTEXT_TOKENS` | `4096` | Context budget; capped at `MODEL_CONTEXT_LENGTH`. |
| `UPLOAD_DIR` | `./storage/uploads` | Relative paths resolve from the backend working directory. |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum upload size. |
| `ALLOWED_EXTENSIONS` | `.pdf` | Comma-separated accepted extensions. |
| `MAX_PDF_PAGES` | `500` | Maximum pages. |
| `MAX_EXTRACTED_TEXT_CHARS` | `5000000` | Maximum extracted text. |
| `MAX_CHUNKS_PER_DOCUMENT` | `10000` | Maximum chunks per document. |
| `ALLOW_PDF_ACTIVE_CONTENT` | `false` | Allow active PDF markers; keep false for untrusted files. |
| `UPLOAD_PROCESS_TIMEOUT_SECONDS` | `180` | Validation/extraction/chunking timeout. |
| `MAX_CONCURRENT_REQUESTS` | `2` | Shared Ollama model semaphore; low-memory profile uses `1`. |
| `MAX_CONCURRENT_UPLOADS` | `2` | Upload concurrency. |
| `MEMORY_PROFILE` | `default` | `default` or `low-memory`; applies smaller defaults only where not explicitly set. |
| `MEMORY_GUARD_ENABLED` | `true` | Check RAM/VRAM before model calls and unload under pressure. |
| `OLLAMA_UNLOAD_ON_MEMORY_PRESSURE` | `true` | Unload this app's models after a request when memory is still tight. |
| `MIN_FREE_MEMORY_MB` | `1024` | Host RAM pressure threshold. |
| `MIN_FREE_VRAM_MB` | `512` | GPU VRAM pressure threshold. |
| `MEMORY_CHECK_INTERVAL_SECONDS` | `30` | Idle-unload monitor interval. |
| `MODEL_CONTEXT_LENGTH` | `4096` | Model context window used for context budgeting; low-memory profile uses `2048`. |
| `BATCH_SIZE` | `16` | Upper bound for embedding batches. |
| `INDEX_BATCH_SIZE` | `8` | Chunks embedded and committed per indexing batch. |
| `MAX_INDEX_BATCH_FAILURES` | `3` | Stop indexing after this many consecutive failed batches. |
| `MAX_GRAPH_LIMIT` | `200` | Maximum graph result/node limit. |
| `MAX_GRAPH_TRAVERSAL_DEPTH` | `3` | Hard graph depth cap. |
| `MAX_GRAPH_EXTRACTION_CHUNKS` | `100` | Chunks processed per graph extraction request. |
| `MAX_EVIDENCE_ITEMS` | `20` | Maximum evidence items returned. |
| `MAX_EVIDENCE_CHARS` | `4000` | Per-evidence character cap. |
| `MAX_QUESTION_CHARS` | `2000` | Chat question length cap. |
| `MAX_TOP_K` | `50` | Hard vector `top_k` cap. |
| `MAX_EXTRACTED_ENTITIES_PER_CHUNK` | `30` | Model-output cap. |
| `MAX_EXTRACTED_RELATIONSHIPS_PER_CHUNK` | `30` | Model-output cap. |
| `GRAPH_EXTRACTION_TIMEOUT_SECONDS` | `600` | Per-request graph extraction timeout. |

## Performance and memory

Measured results for this repository's target laptop (RTX 4050 6 GB, 16 GB RAM)
are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). The short version, measured
with Ollama 0.40.x and `qwen3:4b` on the actual machine:

| Profile | Context / prompt batch | Generation median | Peak VRAM | Embeddings, batch 8 |
| --- | --- | --- | --- | --- |
| Default | 4096 / 512 | 41.5 tok/s | 4,368 MB | 98.6 texts/s |
| Low-memory | 2048 / 128 | 45.8 tok/s | 4,001 MB | 105.54 texts/s |

These are short, warm-up-controlled local measurements, not a capacity plan.
The low-memory run happened under different background load and was slightly
faster on that occasion; the reliable tradeoff is lower context/request memory,
not a guaranteed speed increase. Before the local Ollama CUDA runtime was
repaired, the same machine reported `100% CPU` and measured 7.3 tok/s, which is
why the application checks `/api/ps` instead of assuming GPU acceleration.

Run the benchmark yourself from `backend/`:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_local --json benchmark-report.json
.\.venv\Scripts\python.exe -m scripts.benchmark_local `
    --context 2048 --num-batch 128 --json benchmark-low-memory.json
```

The script samples real `nvidia-smi` values and host RAM, labels derived
estimates separately, and unloads the app's models when finished. It does not
measure full-PDF ingestion or model quality.

To inspect or free memory at runtime:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/status
Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/gpu
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/v1/system/unload
```

The system endpoints only unload models named by this application's
configuration; they do not unload unrelated Ollama models.

## Shutdown and restart

Stop the backend and frontend with `Ctrl+C` in their terminals. Stop the data
services without deleting data:

```powershell
docker compose -f docker-compose.dev.yml stop
```

Start them again with:

```powershell
docker compose -f docker-compose.dev.yml start
```

`docker compose ... down` removes the containers and network but keeps named
volumes. `docker compose ... down -v` also deletes the database/graph data and
must only be used intentionally. The app does not auto-start PostgreSQL, Neo4j,
Ollama, the backend, or the frontend after a reboot.

## Testing and verification

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -m evaluation -v
.\.venv\Scripts\python.exe -m scripts.evaluate
.\.venv\Scripts\python.exe -m scripts.evaluate --json eval-report.json
.\.venv\Scripts\python.exe -m scripts.benchmark_local --json benchmark-report.json
```

From `frontend/`:

```powershell
npm run lint
npx tsc --noEmit
npm run test:e2e
npm run build
```

The deterministic suite exercises production ingestion, retrieval, generation,
citation, and deletion code with doubles only at the embedding model, vector
store, graph store, and LLM boundaries. It also covers resumable indexing,
memory-pressure behavior, measured system endpoints, and low-memory profile
defaults. It does not measure live model quality. Playwright mocks
`/api/v1/chat/` and `/api/v1/graph/`; it verifies rendering, citations, graph
nodes/edges, node details, empty states, and API-failure states, not real model
output.

Optional live suite (requires the running backend, frontend, PostgreSQL,
Neo4j, and Ollama):

```powershell
Set-Location backend
$env:KG_LIVE = "1"
.\.venv\Scripts\python.exe -m pytest tests/integration/test_live_e2e.py -v -s
```

The live suite is slow on CPU-only machines and is intentionally skipped by
default. It includes a controlled real-Neo4j round trip so edge storage and API
readback are covered even when the small generation model extracts no usable
relationship. `backend/scripts/smoke.py` is an additional manual end-to-end
check, but it asserts that the live model extracts at least one relationship and
can therefore fail with a sparse model output.

### Last verified results (2026-10-31, this audit environment)

These are checks actually run on this machine, not guarantees for every
machine. The backend suite and benchmark were re-run on 2026-10-31; the
frontend checks are from the 2026-10-30 audit and were not affected by the
backend-only changes.

| Command | Result |
| --- | --- |
| `python -m pytest -q` | 158 passed, 2 skipped |
| `python -m scripts.evaluate` | 10/10 checks passed; graph precision 0.88, recall 1.00, F1 0.94 |
| `python -m scripts.benchmark_local --json benchmark-report.json` | Default profile: 41.5 tok/s generation median, 4,368 MB peak VRAM |
| `python -m scripts.benchmark_local --context 2048 --num-batch 128` | Low-memory: 45.8 tok/s, 4,001 MB peak VRAM |
| `npm run lint` | Passed (2026-10-30) |
| `npx tsc --noEmit` | Passed (2026-10-30) |
| `npm run test:e2e` | 5 passed (2026-10-30) |
| `npm run build` | Passed, one non-fatal Turbopack package-lock warning (2026-10-30) |
| `pip check` | No broken requirements |
| `/api/v1/health/ready` | `ready: true`; Ollama, Neo4j, PostgreSQL, pgvector available |
| `ollama ps` | `100% GPU` after the CUDA runtime repair |
| `ollama list` | `qwen3:4b` and `nomic-embed-text:latest` present |

## Troubleshooting

### `python` is not found or opens the Microsoft Store

Install a 64-bit Python 3.12+ build and use the actual interpreter path. If the
venv already exists, call `.\.venv\Scripts\python.exe` directly; do not rely on
PowerShell activation.

### `Activate.ps1 cannot be loaded` / execution policy

This is a PowerShell policy issue, not an application error. Skip activation and
run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Port 8000 or 3000 is already in use

Check the listener:

```powershell
Get-NetTCPConnection -LocalPort 8000,3000 -State Listen
```

Stop the unrelated process or start the backend/frontend on another port. If you
change the backend port, also update `NEXT_PUBLIC_API_URL` and restart the
frontend dev server.

### PostgreSQL authentication or missing tables

- The password in `.env` must match the password that initialized the data
  volume. Changing only `.env` does not change an existing database user.
- Migrations only run on first initialization of an empty volume. Apply them
  manually with the `docker exec ... -f /docker-entrypoint-initdb.d/...`
  commands above.
- Check `docker compose ... ps` and `docker logs kg-postgres`.
- If pgvector is missing, verify you are using the `pgvector/pgvector:pg17`
  image, not a stock PostgreSQL image.

### Neo4j authentication failure

The bundled Compose file initializes `NEO4J_AUTH` as `neo4j/<password>` on the
first volume initialization. If you changed `NEO4J_PASSWORD` after that, either
use the original password or intentionally reset the Neo4j volume. Keep
`NEO4J_USER=neo4j` with Compose. Neo4j Browser is at
<http://127.0.0.1:7474>.

### Ollama is unavailable or a model is missing

```powershell
ollama list
ollama pull qwen3:4b
ollama pull nomic-embed-text
Invoke-RestMethod http://127.0.0.1:8000/api/v1/models/status
```

Check that `OLLAMA_HOST` matches the running server (normally
`http://localhost:11434`). The model names must match exactly. If Ollama is
running but the backend cannot see it, check for a firewall/proxy or a stale
backend process with old environment variables.

### If Ollama reports 100% CPU

Symptom: `ollama ps` shows `100% CPU` while a model is loaded, GPU utilization is
near zero, and generation is roughly an order of magnitude slower than expected.
The presence of an NVIDIA GPU is not enough; Ollama must load its CUDA backend.

1. Confirm the driver and GPU are visible:

   ```powershell
   nvidia-smi
   ollama --version
   ollama ps
   ```

2. Inspect the Ollama installation's `lib/ollama/cuda_v12` directory. A complete
   Windows CUDA runtime includes `ggml-cuda.dll`, `cudart64_12.dll`,
   `cublas64_12.dll`, `cublasLt64_12.dll`, and the Visual C++ runtime DLLs. A
   partially extracted install can leave only a temporary file and silently fall
   back to CPU.
3. Repair or reinstall Ollama with the matching official Windows package. If you
   restore files manually, use the complete `cuda_v12` directory from the
   official Ollama Windows archive; do not mix arbitrary CUDA DLL versions.
4. Restart the Ollama application/server and call a model again. Confirm
   `ollama ps` reports `100% GPU`, then check
   `Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/status`.

On the machine used for this project, repairing the incomplete CUDA runtime
changed the measured rate from 7.3 tok/s (CPU) to 41.5 tok/s (GPU). The app
continues to work on CPU, but indexing and graph extraction are much slower.

### Out of memory / HTTP 503

The API returns HTTP 503 when a local model call fails for lack of RAM/VRAM or
when memory is critically low before a call. The message includes the measured
free system RAM/VRAM and suggestions. To recover:

1. Check the current state:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/status
   Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/gpu
   ```

2. Set `MEMORY_PROFILE=low-memory` (or lower `MODEL_CONTEXT_LENGTH`,
   `OLLAMA_MAX_TOKENS`, `OLLAMA_NUM_BATCH`, and batch sizes) and restart the
   backend.
3. Free VRAM/RAM by closing other GPU-heavy applications, or call
   `POST /api/v1/system/unload`.
4. Retry indexing. Completed batches were committed; the endpoint resumes from
   the remaining chunks.

Do not retry in a tight loop while memory is still exhausted. The indexer stops
after `MAX_INDEX_BATCH_FAILURES` consecutive failures and on the first
out-of-memory error.

### Chat says there is insufficient evidence

1. Upload a text-based PDF with extractable text.
2. Call `POST /api/v1/documents/{id}/index`; do not stop at upload.
3. Confirm `indexed == total` and `failed == 0` from the index status endpoint.
4. If you need graph evidence, call `POST /api/v1/graph/extract/{id}` and check
   the returned entity/relationship counts.
5. If the question is outside the indexed corpus, the correct behavior is to
   abstain. Try a document-scoped question or a larger `TOP_K_CHUNKS`.

### Graph Explorer is empty

Graph extraction is a separate step and the model may extract no relationships
for a given document. Check the graph extraction response, then query
`/api/v1/graph/?document_id=...`. The graph is not populated by upload or vector
indexing alone. A sparse or empty graph is a model/extraction limitation, not a
frontend rendering failure.

### Upload is rejected

Check the file is a real PDF, not an encrypted/empty/scanned-only file, is below
`MAX_UPLOAD_SIZE_MB`, has an allowed extension, and does not trigger active-content
rejection. Scanned PDFs require an OCR step that this project does not implement.
If you deliberately trust a local corpus containing active content, only then
consider `ALLOW_PDF_ACTIVE_CONTENT=true`.

### `npm run build` warns about a package-lock outside the repository

Turbopack can notice a parent-directory `package-lock.json` and print a
non-fatal warning. It does not fail the build. If desired, set `turbopack.root`
in `frontend/next.config.ts` to the frontend directory. The audit build passed
with this warning.

### Frontend cannot reach the API

Confirm the backend is running on the host/port in `NEXT_PUBLIC_API_URL`, the
frontend origin is listed in `BACKEND_CORS_ORIGINS`, and the dev server was
restarted after changing the environment. Browser requests go directly to
`http://localhost:8000` in the default configuration.

## Known limitations and incomplete features

- **No authentication or multi-user isolation.** The API trusts any caller that
  can reach it. Keep it on loopback; do not expose it to a network.
- **The UI lacks index/extract controls.** Upload/list/delete, chat, and graph
  viewing exist, but indexing and graph extraction must be triggered through the
  API/Swagger after upload.
- **Dashboard is a placeholder.** The three cards on `/` show static dashes and
  a "Loading..." label; they are not wired to live metrics.
- **Graph retrieval is document-scoped but question-independent.** It can append
  all entities/evidence from the selected document(s), which lowers precision.
  The graph channel is not bounded by vector `top_k`, so evidence counts can be
  larger than the vector window.
- **Live extraction quality is model-dependent.** In the tested environment,
  `qwen3:4b` sometimes extracted very few entities and no usable relationships.
  The deterministic suite and controlled Neo4j round trip verify the plumbing,
  not model accuracy.
- **GPU acceleration depends on the local Ollama runtime.** The app measures
  where Ollama placed the model; if the CUDA libraries are incomplete, Ollama
  falls back to CPU and runs much more slowly. The benchmark is a short local
  throughput check, not a full-PDF ingestion or capacity-planning measurement.
- **No OCR.** Scanned/image-only PDFs fail or produce no useful text.
- **Active-content PDFs are rejected by default**, even if the text would be
  useful.
- **Faithfulness is a lexical proxy** in the evaluation, not entailment or a
  human quality judgment.
- **No cross-store transaction.** Deletion cleans Neo4j first on a best-effort
  basis; a graph failure is reported, and orphan cleanup is available through
  `backend/scripts/cleanup_graph.py`.
- **`ingestion_jobs` is schema-only** in the current synchronous flow. There is
  no background job queue or resumable ingestion worker.
- **No streaming responses or cancellation.** Long generation/extraction
  requests hold the HTTP connection open.
- **`NEO4J_QUERY_TIMEOUT_SECONDS` and `TOP_K_ENTITIES` are configured but not
  currently enforced by the active query code.**
- **Docker Compose assumes Neo4j user `neo4j`** and loopback-only published
  ports. Use the documented settings or adapt the Compose file deliberately.
- **Not production ready.** There is no TLS, auth, rate limiting beyond
  concurrency caps, audit logging, background processing, or deployment
  hardening.

## Security notes

- `.env` is gitignored; only `.env.example` is tracked, with placeholder
  passwords.
- Passwords and connection strings are not returned by health/model endpoints.
- SQL and Cypher values are parameterized. Cypher interpolation is limited to
  clamped integers.
- Uploaded filenames are sanitized and stored names are server-generated UUIDs.
- Uploads are streamed with a hard size cap; PDF validation checks signature,
  page count, extracted-text size, and active content.
- CORS is restricted to explicit origins and wildcards are rejected.
- Model output is schema-validated before graph storage; entity types and
  predicates use approved vocabularies.
- Retrieved document text is placed in an untrusted-evidence block with
  instructions not to follow prompt injections, and citations are checked
  against retrieved evidence.

For the evaluation methodology and current metrics, see
[docs/EVALUATION.md](docs/EVALUATION.md). For test organization and markers, see
[docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md). For measured local
throughput, VRAM/RAM behavior, and low-memory tradeoffs, see
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).
