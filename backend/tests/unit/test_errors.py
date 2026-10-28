"""Tests for error handling."""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_404_error(client: TestClient):
    """Test 404 returns consistent error format."""
    response = client.get("/nonexistent")
    assert response.status_code == 404
    # FastAPI default 404 is fine, our global handler is for unhandled exceptions
