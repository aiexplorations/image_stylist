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

# Import from the main app
from app import generate_image, StyleRequest, initialize_pipeline, system_info

# Initialize the application
app = FastAPI(title="Image Style Transfer")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Reset model endpoint
@app.post("/reset-model")
async def reset_model():
    """Reset the model pipeline to resolve potential MPS/CUDA errors"""
    try:
        # Import torch here since we need it
        import torch
        from app import initialize_pipeline, pipe, model_id, current_device
        
        if pipe is not None:
            print("Moving pipeline to CPU first to clear state")
            pipe = pipe.to("cpu")
            
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                print("MPS cache cleared")
                
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            
            device = current_device
            print(f"Reloading pipeline with device {device}")
            initialize_pipeline(model_id, device)
            
        return {"status": "success", "message": "Model pipeline reset successfully"}
    except Exception as e:
        print(f"Error resetting model: {str(e)}")
        return JSONResponse(
            content={"error": f"Error resetting model: {str(e)}"},
            status_code=500
        )

# Serve the HTML file for the root path
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(current_dir, "ollama_styler.html"), "r") as f:
        return f.read()

# Forward the generate endpoint
@app.post("/generate")
async def generate_proxy(request: Request):
    body = await request.json()
    return await generate_image(StyleRequest(**body))

# System info endpoint
@app.get("/system-info")
def get_system_info_endpoint():
    return system_info()

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
    print("Preloading model...")
    initialize_pipeline()
    
    # Run the server
    port = int(os.environ.get("PORT", 8081))
    print(f"Starting server on port {port}...")
    print(f"Visit http://localhost:{port} in your browser")
    uvicorn.run("app_server:app", host="0.0.0.0", port=port, reload=True)
