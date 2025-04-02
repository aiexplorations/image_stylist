import pytest
from fastapi.testclient import TestClient
from src.api.models import StyleRequest

def test_health_check(test_client: TestClient):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Service is healthy"
    assert "pipeline_initialized" in data["data"]
    assert "current_device" in data["data"]
    assert "device_status" in data["data"]

def test_reset_model(test_client: TestClient, mock_pipeline):
    """Test the model reset endpoint."""
    response = test_client.post("/reset-model")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Model pipeline reset successfully" in data["message"]

def test_generate_image(test_client: TestClient, sample_base64_image, mock_pipeline):
    """Test the image generation endpoint."""
    request_data = {
        "image": sample_base64_image,
        "prompt": "test prompt",
        "strength": 0.75,
        "steps": 30,
        "device": "cpu"
    }
    
    response = test_client.post("/generate", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Image generated successfully"
    assert "image" in data["data"]
    assert data["data"]["image"].startswith("data:image/jpeg;base64,")

def test_generate_image_invalid_base64(test_client: TestClient):
    """Test the image generation endpoint with invalid base64 image."""
    request_data = {
        "image": "invalid_base64",
        "prompt": "test prompt",
        "strength": 0.75,
        "steps": 30,
        "device": "cpu"
    }
    
    response = test_client.post("/generate", json=request_data)
    assert response.status_code == 500
    assert "Invalid base64 image" in response.json()["detail"]

def test_generate_image_missing_prompt(test_client: TestClient, sample_base64_image):
    """Test the image generation endpoint with missing prompt."""
    request_data = {
        "image": sample_base64_image,
        "strength": 0.75,
        "steps": 30,
        "device": "cpu"
    }
    
    response = test_client.post("/generate", json=request_data)
    assert response.status_code == 422  # Validation error 