"""Tests for health endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "KnowledgeGraph AI API"
    assert data["local_only"] is True


@pytest.mark.unit
def test_health_endpoint(client: TestClient):
    """Test /api/v1/health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data
    assert data["services"]["api"]["status"] == "healthy"


@pytest.mark.unit
def test_readiness_endpoint(client: TestClient):
    """Test /api/v1/health/ready endpoint."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ready" in data
    assert "dependencies" in data
    # Should not expose credentials
    assert "password" not in str(data).lower()
    assert "secret" not in str(data).lower()
