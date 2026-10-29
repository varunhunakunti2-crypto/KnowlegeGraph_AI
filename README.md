# KnowledgeGraph AI

A fully local GraphRAG document assistant for Windows with RTX 4050 6GB VRAM and 16GB RAM. Upload PDFs, build a knowledge graph, and ask questions with citations - all running locally with no paid APIs.

## Stack

- **Frontend:** Next.js App Router, TypeScript, Tailwind CSS, React Flow
- **Backend:** FastAPI (Python 3.12), REST API
- **Local AI:** Ollama (qwen3:4b for generation, nomic-embed-text for embeddings)
- **Graph Database:** Neo4j Community Edition
- **Vector Store:** PostgreSQL + pgvector
- **PDF Processing:** PyMuPDF
- **Testing:** pytest (backend), Playwright (frontend display)

## Architecture

```
PDF Upload → Validation → Text Extraction (PyMuPDF) → Chunking
    ↓
Embeddings (nomic-embed-text via Ollama) + Entity/Relation Extraction (qwen3:4b)
    ↓
PostgreSQL + pgvector (chunks + embeddings) | Neo4j (entities/relations/graph)
    ↓
Hybrid Retrieval (vector similarity + graph traversal) → Context Assembly
    ↓
Answer Generation (qwen3:4b via Ollama) → Citations (doc/chunk/page)
```

## Quick Start (Local Setup)

### Prerequisites

- Windows 10/11
- Python 3.12
- Node.js 18+ (for Next.js)
- [Ollama](https://ollama.com/download)
- [Neo4j Community Edition](https://neo4j.com/download-center/#community)
- [PostgreSQL](https://www.postgresql.org/download/windows/) with [pgvector](https://github.com/pgvector/pgvector)

### 1. Pull Ollama models

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env with your local database credentials
```

### 3. Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 5. Database Setup

- Create PostgreSQL database `knowledgegraph` and enable pgvector extension
- Start Neo4j Community Edition locally

## Project Structure

```
KnowledgeGraph AI/
├── frontend/          # Next.js App Router UI
├── backend/           # FastAPI app (modular layers)
│   ├── app/
│   │   ├── apis/      # API routes
│   │   ├── services/  # Business logic
│   │   ├── ingestion/ # PDF upload/validation/chunking
│   │   ├── extraction/# Entity/relation extraction
│   │   ├── graph/     # Neo4j graph operations
│   │   ├── retrieval/ # Hybrid retrieval
│   │   ├── generation/# Answer generation with citations
│   │   └── databases/ # DB connections/schemas
├── database/          # Migrations and schema definitions
├── scripts/           # Setup/dev utility scripts
├── storage/           # Local uploads and artifacts (gitignored)
└── docs/              # Documentation
```

## Key Principles

- **Fully local** - All inference via Ollama, no external API calls
- **GraphRAG** - Combines vector similarity with knowledge graph traversal
- **Citations** - Every answer includes source document, chunk, and page references
- **Modular** - Clear separation of concerns across layers
- **Resource-conscious** - Tuned for RTX 4050 6GB VRAM, 16GB RAM
- **Safe ingestion** - Idempotent, preserves metadata, validates inputs

## Hardware Constraints

Target: Windows, NVIDIA RTX 4050 (6GB VRAM), 16GB RAM. Model sizes and concurrency are configured conservatively. GPU acceleration is not assumed - verify with your setup.

## Testing

- [docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) - test organization, markers, and commands
- [docs/EVALUATION.md](docs/EVALUATION.md) - deterministic GraphRAG evaluation, metrics, vector-only comparison, Playwright display tests, and the optional live suite

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q          # deterministic suite
.\.venv\Scripts\python.exe -m scripts.evaluate   # evaluation report

cd ..\frontend
npm run test:e2e                                 # browser display tests
```
