"""System status endpoints expose measured values, not assumptions."""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_system_status_returns_measured_snapshot(client: TestClient, monkeypatch):
    import app.api.v1.system as system_api

    monkeypatch.setattr(
        system_api.memory,
        "runtime_status",
        lambda: {
            "profile": "low-memory",
            "gpu": {"available": True, "name": "RTX 4050", "free_mb": 1500},
            "system_memory": {"available": True, "available_mb": 2048},
            "pressure": {"active": False, "reasons": []},
            "limits": {"index_batch_size": 4},
        },
    )

    response = client.get("/api/v1/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "low-memory"
    assert body["gpu"]["free_mb"] == 1500
    assert body["limits"]["index_batch_size"] == 4


@pytest.mark.unit
def test_system_gpu_reports_when_nvidia_smi_is_unavailable(client: TestClient, monkeypatch):
    import app.api.v1.system as system_api

    monkeypatch.setattr(system_api.memory, "gpu_status", lambda force=False: None)
    response = client.get("/api/v1/system/gpu")
    assert response.status_code == 200
    assert response.json()["measured"] is False
    assert "nvidia-smi" in response.json()["note"]


@pytest.mark.unit
def test_system_unload_only_unloads_app_models(client: TestClient, monkeypatch):
    import app.api.v1.system as system_api

    monkeypatch.setattr(
        system_api.memory, "unload_app_models", lambda: ["qwen3:4b"]
    )
    response = client.post("/api/v1/system/unload")
    assert response.status_code == 200
    assert response.json()["unloaded"] == ["qwen3:4b"]


@pytest.mark.unit
def test_index_endpoint_returns_503_with_resume_guidance(client: TestClient, monkeypatch):
    import app.api.v1.documents as documents
    from app.core.memory import InsufficientMemoryError

    monkeypatch.setattr(documents.DocumentDB, "document_exists", lambda _: True)

    async def out_of_memory(_document_id):
        raise InsufficientMemoryError(
            "Out of memory during embedding. Indexing is resumable: 2 chunks "
            "were committed."
        )

    monkeypatch.setattr(documents.VectorDB, "index_document_chunks", out_of_memory)
    response = client.post(
        "/api/v1/documents/123e4567-e89b-12d3-a456-426614174000/index"
    )

    assert response.status_code == 503
    assert "resumable" in response.json()["detail"].lower()
