import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from src.api.routes import app
from src.core.pipeline import PipelineManager

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.fixture
def pipeline_manager():
    """Create a PipelineManager instance for testing."""
    return PipelineManager()

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple gradient image
    img = Image.new('RGB', (512, 512))
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = (i % 256, j % 256, 100)
    return img

@pytest.fixture
def sample_base64_image(sample_image):
    """Create a base64-encoded sample image."""
    buffered = BytesIO()
    sample_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@pytest.fixture
def mock_pipeline(monkeypatch):
    """Create a mock Stable Diffusion pipeline for testing."""
    class MockPipeline:
        def __init__(self):
            self.device = "cpu"
            
        def to(self, device):
            self.device = device
            return self
            
        def __call__(self, **kwargs):
            # Return a mock result with a sample image
            class MockResult:
                def __init__(self):
                    self.images = [Image.new('RGB', (512, 512), color='red')]
            return MockResult()
            
        def enable_attention_slicing(self):
            pass
    
    def mock_from_pretrained(*args, **kwargs):
        return MockPipeline()
    
    monkeypatch.setattr(
        "diffusers.StableDiffusionImg2ImgPipeline.from_pretrained",
        mock_from_pretrained
    )
    return MockPipeline() 