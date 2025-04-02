import platform
import torch
import os
from typing import Tuple
import logging

def setup_device() -> tuple[str, bool]:
    """Detects available device (MPS, CUDA, CPU) and returns device name and apple_silicon flag."""
    force_cpu = os.environ.get("FORCE_CPU_FLAG", "false").lower() == "true"
    
    device = "cpu"
    is_apple_silicon = False

    if force_cpu:
        logging.warning("CPU usage forced via --cpu flag.")
        device = "cpu"
        # Check if it *is* Apple Silicon, even if forced to CPU
        is_apple_silicon = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logging.info("Metal Performance Shaders (MPS) is available!")
        device = "mps"
        is_apple_silicon = True
    elif torch.cuda.is_available():
        logging.info("CUDA is available.")
        device = "cuda"
    else:
        logging.info("MPS and CUDA not available, using CPU.")
        device = "cpu"
        
    logging.info(f"Device selected for use: {device}")
    return device, is_apple_silicon

def is_mps_working() -> bool:
    """Quick check if MPS is available."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

def clear_device_memory(device: str):
    """Clears appropriate cache based on the device."""
    try:
        if device == "mps":
            torch.mps.empty_cache()
            logging.info("Cleared MPS memory cache.")
        elif device == "cuda":
            torch.cuda.empty_cache()
            logging.info("Cleared CUDA memory cache.")
        else:
            # No specific cache clear needed for CPU in this context
            pass 
    except Exception as e:
        logging.warning(f"Could not clear device memory for {device}: {e}") 