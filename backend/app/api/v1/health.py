"""Health check endpoints."""
from fastapi import APIRouter, status
from app.schemas.health import HealthResponse, ReadinessResponse
from app.core.config import settings
import httpx
from neo4j import GraphDatabase
import psycopg
from typing import Dict, Any

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> Dict[str, Any]:
    """Basic health check."""
    return {
        "status": "ok",
        "services": {
            "api": {"status": "healthy"}
        }
    }


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness() -> Dict[str, Any]:
    """
    Readiness check - reports dependency status without exposing credentials.
    """
    dependencies = {
        "ollama": {"status": "unknown"},
        "neo4j": {"status": "unknown"},
        "postgres": {"status": "unknown"},
    }
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                dependencies["ollama"]["status"] = "ready"
                try:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    dependencies["ollama"]["models"] = {
                        "qwen3_4b": any("qwen3:4b" in m for m in models),
                        "nomic_embed": any("nomic-embed" in m for m in models),
                    }
                except Exception:
                    dependencies["ollama"]["models"] = "unknown"
            else:
                dependencies["ollama"]["status"] = "not_ready"
    except Exception:
        dependencies["ollama"]["status"] = "unreachable"
    
    # Check Neo4j
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run("RETURN 1").single()
        driver.close()
        dependencies["neo4j"]["status"] = "ready"
    except Exception:
        dependencies["neo4j"]["status"] = "unreachable"
    
    # Check PostgreSQL
    try:
        conn = psycopg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            connect_timeout=3,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
            # Check pgvector
            try:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                if cur.fetchone():
                    dependencies["postgres"]["pgvector"] = "available"
                else:
                    dependencies["postgres"]["pgvector"] = "missing"
            except Exception:
                dependencies["postgres"]["pgvector"] = "unknown"
        conn.close()
        dependencies["postgres"]["status"] = "ready"
    except Exception:
        dependencies["postgres"]["status"] = "unreachable"
    
    all_ready = all(d.get("status") == "ready" for d in dependencies.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "ready": all_ready,
        "dependencies": dependencies,
    }
