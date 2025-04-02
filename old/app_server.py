"""
Direct Standalone Image Style Transfer App
This version combines the FastAPI backend with static file serving
for a simpler development and testing setup.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import uvicorn
import time

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import from the new modular structure
from src.api.models import StyleRequest
from src.api.routes import app, generate_image
from src.core.pipeline import PipelineManager
from src.core.device import setup_device

# Initialize pipeline manager
pipeline_manager = PipelineManager()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reset model endpoint
@app.post("/reset-model")
async def reset_model():
    """Reset the model pipeline to resolve potential MPS/CUDA errors"""
    try:
        result = pipeline_manager.reset_pipeline()
        if result["status"] == "error":
            return JSONResponse(
                content={"error": result["message"]},
                status_code=500
            )
        return result
    except Exception as e:
        print(f"Error resetting model: {str(e)}")
        return JSONResponse(
            content={"error": f"Error resetting model: {str(e)}"},
            status_code=500
        )

# Serve the HTML file for the root path
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(current_dir, "src/static/index.html"), "r") as f:
        return f.read()

# Forward the generate endpoint
@app.post("/generate")
async def generate_proxy(request: Request):
    try:
        print("Received generate request...")
        body = await request.json()
        print(f"Processing with prompt: {body.get('prompt', 'No prompt provided')}")
        start_time = time.time()
        
        response = await generate_image(StyleRequest(**body))
        
        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")
        return response
    except Exception as e:
        print(f"Error in generate_proxy: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e), "data": None},
            status_code=500
        )

# System info endpoint
@app.get("/system-info")
def get_system_info_endpoint():
    device, is_apple_silicon = setup_device()
    return {
        "status": "ok",
        "device": device,
        "is_apple_silicon": is_apple_silicon,
        "pipeline_initialized": pipeline_manager.is_initialized()
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Image Style Transfer API is running"}

# Debug endpoint
@app.get("/debug-image")
async def debug_image():
    """Return the debug image if it exists"""
    debug_path = "/tmp/debug_output.jpg"
    if os.path.exists(debug_path):
        return FileResponse(debug_path)
    else:
        return JSONResponse(
            content={"error": "No debug image available"},
            status_code=404
        )
        
# Style guide endpoint
@app.get("/style_prompt_guide.md", response_class=HTMLResponse)
async def style_guide():
    """Render the style guide as HTML"""
    try:
        with open(os.path.join(current_dir, "style_prompt_guide.md"), "r") as f:
            content = f.read()
            
        # Simple markdown to HTML conversion for basic display
        content = content.replace("\n\n", "<br><br>")
        content = content.replace("# ", "<h1>") + "</h1>"
        content = content.replace("## ", "<h2>") 
        content = content.replace("\n## ", "<br><h2>")
        content = content.replace("</h1>\n", "</h1>")
        content = content.replace("\n<h2>", "</h2><h2>")
        content = content.replace("- ", "<li>")
        content = content.replace("\n-", "</li>\n<li>")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Style Transfer Prompt Guide</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; max-width: 800px; margin: 0 auto; }}
                h1 {{ color: #2d3748; margin-top: 20px; }}
                h2 {{ color: #4a5568; margin-top: 25px; }}
                li {{ margin-bottom: 8px; }}
                code {{ background-color: #f7fafc; padding: 2px 4px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"<html><body><h1>Error loading style guide</h1><p>{str(e)}</p></body></html>"

if __name__ == "__main__":
    # Initialize the model at startup
    print("\n=== Image Style Transfer Application Starting ===")
    print("Detecting hardware configuration...")
    device, is_apple = setup_device()
    print(f"• Device selected: {device}")
    print(f"• Apple Silicon: {'Yes' if is_apple else 'No'}")
    
    print("\nPreloading model pipeline...")
    try:
        pipeline_manager.initialize_pipeline("runwayml/stable-diffusion-v1-5", device)
        print("• Model pipeline initialized successfully")
        print(f"• Current device: {pipeline_manager.current_device}")
        print(f"• Model ID: {pipeline_manager.model_id}")
    except Exception as e:
        print(f"Warning: Error initializing pipeline: {e}")
        print("Application will attempt to initialize pipeline on first request")
    
    # Run the server
    port = int(os.environ.get("PORT", 8081))
    print(f"\nStarting server on port {port}...")
    print(f"• API URL: http://localhost:{port}")
    print(f"• Web UI: http://localhost:{port}")
    print("=== Startup Complete ===\n")
    uvicorn.run("app_server:app", host="0.0.0.0", port=port, reload=True)
