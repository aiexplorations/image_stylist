from pydantic import BaseModel
from typing import Optional

class StyleRequest(BaseModel):
    """Request model for image styling."""
    image: str  # base64 encoded image
    prompt: str
    model: Optional[str] = "runwayml/stable-diffusion-v1-5"
    strength: Optional[float] = 0.65  # Default to a moderate effect (0.0-1.0)
    steps: Optional[int] = 50        # Number of inference steps
    device: Optional[str] = "auto"   # "auto", "cpu", "mps" (Apple Silicon), or "cuda" (NVIDIA)

class ModelResponse(BaseModel):
    """Response model for API endpoints."""
    status: str
    message: str
    data: Optional[dict] = None 