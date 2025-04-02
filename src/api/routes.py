from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from .models import StyleRequest, ModelResponse
from ..core.pipeline import PipelineManager
from ..core.device import setup_device, is_mps_working
from ..utils.image import decode_base64_image, encode_pil_to_base64, prepare_image_for_model
import time
import os
import logging

app = FastAPI()
pipeline_manager = PipelineManager()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# Correctly calculate path relative to this file (src/api/routes.py) -> ../../static
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.on_event("startup")
async def startup_event():
    """Initialize the model pipeline when the application starts."""
    logging.info("Application startup: Initializing model pipeline...")
    try:
        device, _ = setup_device()
        # Skip dummy inference to avoid startup failures but still initialize the model
        pipeline_manager.initialize_pipeline(pipeline_manager.model_id, device, skip_dummy_inference=True)
        logging.info(f"Startup successful with device: {device}")
    except Exception as e:
        # Error is logged verbosely within initialize_pipeline
        logging.error(f"!!! CRITICAL STARTUP ERROR: Pipeline initialization failed: {e}")
        # Log error but continue with app startup - we'll initialize on-demand
        logging.warning("Continuing app startup despite pipeline initialization failure")
        # Don't re-raise the exception to allow the app to start anyway

@app.post("/reset-model")
async def reset_model() -> ModelResponse:
    """Reset the model pipeline to resolve potential device errors."""
    result = pipeline_manager.reset_pipeline()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return ModelResponse(**result)

@app.post("/generate")
async def generate_image(request: StyleRequest) -> ModelResponse:
    """Generate a styled image based on the input image and prompt."""
    print("\n--- Received image generation request ---")
    print(f"Prompt: {request.prompt[:50]}...") # Log truncated prompt
    print(f"Strength: {request.strength}, Steps: {request.steps}")
    start_time = time.time()
    try:
        # Check if pipeline needs initializing (e.g., if startup failed)
        if not pipeline_manager.is_initialized():
            logging.warning("Pipeline not initialized. Attempting initialization on demand...")
            device, _ = setup_device()
            # This will raise an error if it fails, handled by the outer try/except
            try:
                pipeline_manager.initialize_pipeline(request.model or pipeline_manager.model_id, device, skip_dummy_inference=False)
                logging.info(f"On-demand pipeline initialization successful on {pipeline_manager.current_device}")
            except Exception as init_error:
                # On Apple Silicon, we still want to try using the model even if dummy inference fails
                if device == "mps" and "Dummy inference failed" in str(init_error):
                    logging.warning(f"Continuing despite dummy inference error on MPS: {init_error}")
                else:
                    # For other errors, re-raise
                    raise
        
        print("Decoding base64 image...")
        input_image = decode_base64_image(request.image)
        print(f"Input image decoded: Size {input_image.size}")
        
        print("Preparing image for model...")
        prepared_image = prepare_image_for_model(input_image)
        print(f"Image prepared: Size {prepared_image.size}") # Should usually be 512x512 or similar

        # Generate image
        logging.info("Getting pipeline instance...") # Changed log message
        pipe = pipeline_manager.get_pipeline()
        if pipe is None:
            logging.error("Get pipeline returned None. Last error: %s", pipeline_manager.last_error)
            # Try resetting and reinitializing the pipeline if None
            if "mps" in str(pipeline_manager.current_device).lower():
                logging.warning("Attempting to reset MPS pipeline...")
                device, _ = setup_device()
                pipeline_manager.initialize_pipeline(request.model or pipeline_manager.model_id, device, skip_dummy_inference=True)
                pipe = pipeline_manager.get_pipeline()
                if pipe is None:
                    error_detail = pipeline_manager.last_error or "Model pipeline not available after reset attempt."
                    raise HTTPException(status_code=500, detail=error_detail)
            else:
                error_detail = pipeline_manager.last_error or "Model pipeline not available or failed to initialize."
                raise HTTPException(status_code=500, detail=error_detail)
        
        print(f"Generating image with {request.steps} steps and strength {request.strength}...")
        result = pipe(
            prompt=request.prompt,
            image=prepared_image,
            strength=request.strength,
            num_inference_steps=request.steps
        )
        generation_time = time.time() - start_time
        print(f"Image generation completed in {generation_time:.2f} seconds.")

        # Process and encode output image
        print("Processing and encoding output image...")
        output_image = result.images[0]
        
        # Apply post-processing: remove black bands and increase resolution
        from ..utils.image import process_generated_image
        output_image = process_generated_image(
            output_image,
            remove_black_bands=True,
            min_dimension=1280
        )
        print(f"Processed output image size: {output_image.size}")
        
        base64_output = encode_pil_to_base64(output_image)
        print(f"Output image encoded: Size {output_image.size}, Base64 length: {len(base64_output)}")

        total_time = time.time() - start_time
        print(f"--- Generation request completed successfully in {total_time:.2f}s ---\n")
        return ModelResponse(
            status="success",
            message="Image generated successfully",
            data={"image": base64_output}
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        end_time = time.time()
        logging.error(f"!!! Error during image generation after {end_time - start_time:.2f}s: {type(e).__name__}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        logging.error(f"--- Generation request failed ---\n")
        # Return a more informative error based on the actual exception
        raise HTTPException(status_code=500, detail=f"Generation Error: {type(e).__name__}: {str(e)}")

@app.get("/health")
async def health_check() -> ModelResponse:
    """Check the health status of the API and model pipeline."""
    try:
        is_pipeline_ready = pipeline_manager.is_initialized()
        device_status = "mps_working" if is_mps_working() else "cpu"
        
        return ModelResponse(
            status="success",
            message="Service is healthy",
            data={
                "pipeline_initialized": is_pipeline_ready,
                "current_device": pipeline_manager.current_device,
                "device_status": device_status
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-info")
async def system_info():
    """Return detailed system information for the frontend."""
    import torch
    import platform
    import sys
    
    try:
        # Get PyTorch version
        pytorch_version = torch.__version__
        
        # Check if we're on Apple Silicon
        is_apple_silicon = platform.processor() == 'arm' or 'arm64' in platform.processor()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Pipeline status
        is_pipeline_ready = pipeline_manager.is_initialized()
        
        return JSONResponse({
            "platform": f"{platform.system()}-{platform.machine()}",
            "python_version": platform.python_version(),
            "pytorch_version": pytorch_version,
            "hardware": {
                "processor": platform.processor(),
                "is_apple_silicon": is_apple_silicon,
                "mps_available": mps_available
            },
            "model_info": {
                "pipeline_initialized": is_pipeline_ready,
                "current_device": pipeline_manager.current_device or "not_initialized"
            }
        })
    except Exception as e:
        logging.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=f"System info error: {str(e)}") 