"""Repository secret hygiene and concurrency-limit tests."""
import asyncio
import subprocess
from pathlib import Path

import pytest

from app.core.resources import LazySemaphore

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_env_file_is_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".env"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode == 128:
        pytest.skip("git is not available in this test environment")
    assert result.returncode == 0, ".env must never be tracked by git"


@pytest.mark.unit
def test_env_example_has_only_placeholders():
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "changeme" in example
    # A real credential would not be the documented placeholder.
    for line in example.splitlines():
        stripped = line.strip()
        if stripped.startswith("POSTGRES_PASSWORD=") or stripped.startswith("NEO4J_PASSWORD="):
            assert stripped.endswith("=changeme")


@pytest.mark.unit
def test_gitignore_covers_common_secret_files():
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!.env.example" in ignore
    assert "*.pem" in ignore
    assert "*.key" in ignore


@pytest.mark.unit
async def test_lazy_semaphore_bounds_concurrency():
    semaphore = LazySemaphore(2)
    active = 0
    maximum = 0

    async def worker():
        nonlocal active, maximum
        async with semaphore:
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(worker() for _ in range(6)))
    assert maximum <= 2


def test_generation_and_embeddings_share_one_model_budget():
    """A single local Ollama host must not get 2x the concurrency."""
    from app.core.resources import model_semaphore
    from app.services.embeddings import embedding_service
    from app.services.llm import llm_service

    assert llm_service.concurrency_limit is model_semaphore
    assert embedding_service.concurrency_limit is model_semaphore
