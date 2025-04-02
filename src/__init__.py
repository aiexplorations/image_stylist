"""
Image Stylist - A FastAPI service for image styling using Stable Diffusion.
"""

from .api.routes import app
from .core.pipeline import PipelineManager
from .core.device import setup_device, is_mps_working
from .utils.image import (
    decode_base64_image,
    encode_pil_to_base64,
    prepare_image_for_model,
    process_generated_image
)

__version__ = "1.0.0"
__all__ = [
    "app",
    "PipelineManager",
    "setup_device",
    "is_mps_working",
    "decode_base64_image",
    "encode_pil_to_base64",
    "prepare_image_for_model",
    "process_generated_image"
]
