#!/usr/bin/env python3
"""
Tests for the Image Style Transfer application.
Run with pytest: python -m pytest test_app.py -v
"""

import os
import base64
import pytest
from io import BytesIO
from PIL import Image
import torch
from fastapi.testclient import TestClient

# Import the application
from app import app, base64_to_image, image_to_base64, system_info
from app_server import get_system_info_endpoint

# Create a test client
client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Image Style Transfer API is running" in response.json().get("message", "")

def test_system_info():
    """Test the system info function directly."""
    info = system_info()
    assert "platform" in info
    assert "python_version" in info
    assert "pytorch_version" in info
    assert "hardware" in info
    assert "model_info" in info

def test_base64_image_conversion():
    """Test the base64 image conversion."""
    # Create a test image
    img = Image.new("RGB", (100, 100), color="red")
    
    # Convert to base64
    base64_str = image_to_base64(img)
    assert base64_str.startswith("data:image/jpeg;base64,")
    
    # Convert back to image
    decoded_img = base64_to_image(base64_str)
    assert decoded_img.size == (100, 100)
    
    # Check that the color is approximately the same (allowing for JPEG compression)
    r, g, b = decoded_img.getpixel((50, 50))
    assert r > 200  # Red should be high in a red image

def test_empty_input_validation():
    """Test validation for empty input."""
    response = client.post(
        "/generate",
        json={
            "image": "",  # Empty image
            "prompt": "test",
            "model": "runwayml/stable-diffusion-v1-5"
        }
    )
    # Check that it's a 400 Bad Request with the appropriate error message
    assert response.status_code == 400, f"Expected 400 status code, got {response.status_code}. Response: {response.text}"
    assert "Invalid image data" in response.json().get("detail", ""), f"Response detail: {response.json().get('detail', '')}"

def create_test_image():
    """Create a test image for generation testing."""
    img = Image.new("RGB", (512, 512), color="blue")
    # Add some shapes to make it more interesting
    for i in range(0, 512, 50):
        for j in range(0, 512, 50):
            color = (i % 256, j % 256, (i + j) % 256)
            for x in range(i, min(i + 40, 512)):
                for y in range(j, min(j + 40, 512)):
                    img.putpixel((x, y), color)
    
    # Save to buffer and convert to base64
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

@pytest.mark.skipif(not torch.backends.mps.is_available(),
                   reason="Requires Apple Silicon GPU for this test")
def test_generation_minimal():
    """Test minimal generation without actually running the model (too slow for tests)."""
    # Skip the actual image generation - just test the request validation
    from app import StyleRequest
    
    # Create a valid request
    request = StyleRequest(
        image=create_test_image(),
        prompt="test style",
        strength=0.3,
        steps=10,
        device="cpu"  # Force CPU for tests
    )
    
    # Check request validation
    assert request.image is not None
    assert request.prompt == "test style"
    assert request.strength == 0.3
    assert request.steps == 10
    assert request.device == "cpu"

def test_debug_image_endpoint():
    """Test the debug image endpoint."""
    # Create a test image
    img = Image.new("RGB", (100, 100), color="green")
    img.save("/tmp/debug_output.jpg")
    
    # Import using the app_server version
    from app_server import app as server_app
    test_client = TestClient(server_app)
    
    # Test the endpoint
    response = test_client.get("/debug-image")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    
    # Clean up
    os.remove("/tmp/debug_output.jpg")

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-v", __file__])
