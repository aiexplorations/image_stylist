from typing import Optional, Dict, Any
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import logging as diffusers_logging
import logging
import time
import traceback
from PIL import Image
from .device import clear_device_memory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
diffusers_logging.set_verbosity_error() # Reduce diffusers verbosity

class PipelineManager:
    def __init__(self):
        self.pipe: Optional[StableDiffusionImg2ImgPipeline] = None
        self.model_id: str = "runwayml/stable-diffusion-v1-5"
        self.current_device: str | None = None
        self.last_error: str | None = None
        self.last_fallback_time: float = 0
        logging.info("PipelineManager initialized.")

    def initialize_pipeline(self, model_id: str, device: str, skip_dummy_inference: bool = False):
        """Initialize or re-initialize the diffusion pipeline."""
        self.model_id = model_id
        self.current_device = device
        self.last_error = None
        self.pipe = None # Ensure pipeline is reset before attempting init
        
        logging.info(f"Attempting to initialize pipeline: Model={self.model_id}, Device={self.current_device}")
        start_time = time.time()
        try:
            # Clear cache if using MPS to potentially resolve issues
            if self.current_device == "mps":
                torch.mps.empty_cache()
                logging.info("Cleared MPS cache before initialization.")
            elif self.current_device == "cuda":
                torch.cuda.empty_cache()
                logging.info("Cleared CUDA cache before initialization.")
                
            # For CPU, sometimes it's beneficial to use float32
            # For MPS/CUDA, use float16 for better performance and memory usage
            torch_dtype = torch.float32 if self.current_device == 'cpu' else torch.float16
            
            logging.info(f"Loading model with dtype: {torch_dtype}")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,  # Disable safety checker for demo purposes
                local_files_only=False  # Allow downloading if needed
            ).to(self.current_device)

            end_time = time.time()
            logging.info(f"Pipeline initialized successfully in {end_time - start_time:.2f} seconds.")
            
            # Only run the dummy inference check if not skipped
            if not skip_dummy_inference:
                try:
                    self._run_dummy_inference()
                except Exception as dummy_error:
                    logging.warning(f"Dummy inference failed but continuing: {dummy_error}")
                    # Don't re-raise dummy inference errors - just log them
                    # This allows the app to start even if dummy inference fails
            else:
                logging.info("Dummy inference check skipped as requested.")

        except Exception as e:
            end_time = time.time()
            self.last_error = f"Failed to initialize pipeline: {type(e).__name__}: {e}"
            logging.error(f"!!! Pipeline initialization failed after {end_time - start_time:.2f}s: {self.last_error}")
            logging.error(traceback.format_exc()) # Log full traceback
            self.pipe = None # Ensure pipeline is None if init failed
            self.current_device = None # Reset device if failed
            # Re-raise the exception to signal failure clearly upstream
            raise RuntimeError(self.last_error) from e 

    def _run_dummy_inference(self):
        """Runs a minimal inference to check if the pipeline is functional after load."""
        if not self.pipe:
            logging.warning("Skipping dummy inference: Pipeline is not loaded.")
            return
            
        logging.info("Running dummy inference check...")
        try:
            # Create a larger dummy image that won't result in 0-sized tensors
            dummy_image = Image.new('RGB', (768, 768), color = 'red')
            # Use minimal steps to speed up the test
            _ = self.pipe(
                prompt="test", 
                image=dummy_image, 
                strength=0.3,  # Higher strength to avoid empty tensors 
                num_inference_steps=2,  # At least 2 steps
                guidance_scale=7.5,  # Default guidance scale
                output_type="pil"  # Ensure PIL output for checks
            )
            logging.info("Dummy inference check successful.")
        except Exception as e:
            self.last_error = f"Dummy inference failed: {type(e).__name__}: {e}"
            logging.error(f"!!! Dummy inference check failed: {self.last_error}")
            logging.error(traceback.format_exc()) # Log full traceback
            
            # MPS-specific issue - don't fail completely on dummy inference issues
            # Just log the error but keep the pipeline available
            if self.current_device == "mps":
                logging.warning("MPS device detected - continuing despite dummy inference failure")
                return  # Return without raising exception
            
            # For non-MPS devices, treat dummy failure as fatal
            self.pipe = None 
            self.current_device = None
            # Raise an error to indicate the pipeline isn't working
            raise RuntimeError(self.last_error) from e

    def reset_pipeline(self) -> Dict[str, str]:
        """Reset the model pipeline to resolve potential device errors."""
        try:
            if self.pipe is not None:
                print("Moving pipeline to CPU first to clear state")
                self.pipe = self.pipe.to("cpu")
                clear_device_memory(self.current_device)
                
                print(f"Reloading pipeline with device {self.current_device}")
                self.initialize_pipeline(self.model_id, self.current_device)
                
            return {"status": "success", "message": "Model pipeline reset successfully"}
        except Exception as e:
            print(f"Error resetting model: {str(e)}")
            return {"status": "error", "message": f"Error resetting model: {str(e)}"}

    def get_pipeline(self) -> Optional[StableDiffusionImg2ImgPipeline]:
        """Get the current pipeline instance."""
        return self.pipe

    def is_initialized(self) -> bool:
        """Check if the pipeline is initialized."""
        return self.pipe is not None 