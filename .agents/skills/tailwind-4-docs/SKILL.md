# KnowledgeGraph AI — Agent Instructions

Before performing any implementation task, read:

- rules/PR.md
- rules/TR.md
- rules/SECURITY.md
- rules/TESTING.md
- rules/DOCUMENTATION.md

All five rule files are mandatory and apply to every task.

## Project Architecture
- Next.js + TypeScript frontend
- FastAPI + Python backend
- Ollama: qwen3:4b
- Embeddings: nomic-embed-text
- Neo4j Community Edition for knowledge graphs
- PostgreSQL + pgvector for vector search
- PyMuPDF for PDF extraction
- React Flow for graph visualization

## Required Behavior
1. Inspect existing code before editing.
2. Follow the five rule files.
3. Identify affected modules before implementation.
4. Make focused changes without unnecessary rewrites.
5. Add appropriate tests.
6. Run relevant checks and report actual results.
7. Update documentation when necessary.
8. Report changed files, failures, and incomplete work.
9. Do not claim completion without verification.
10. If instructions conflict, explain the conflict before making a major architectural change.

Always Use:
- tailwind-4-docs, web-design-guidelines these 2 skills for this project
- DESIGN.md for this project design

These instructions apply to every implementation phase of KnowledgeGraph AI.