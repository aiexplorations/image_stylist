#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Unified run script for Image Style Transfer application
# This script handles setup, dependency installation, and launching the application

echo "====== Image Style Transfer ======"

# Parse command line options
MODE="run"
FORCE_CPU=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      MODE="clean"
      shift
      ;;
    --test)
      MODE="test"
      shift
      ;;
    --cpu)
      FORCE_CPU=true
      shift
      ;;
    --help)
      echo "Usage: ./run.sh [OPTIONS]"
      echo ""
      echo "OPTIONS:"
      echo "  --clean      Remove cache and temporary files"
      echo "  --test       Run tests"
      echo "  --cpu        Force CPU usage (ignore MPS/GPU)"
      echo "  --help       Show this help message"
      echo ""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './run.sh --help' for usage information"
      exit 1
      ;;
  esac
done

# Clean mode: remove cache and temporary files
if [[ "$MODE" == "clean" ]]; then
    echo "🧹 Cleaning up temporary files and cache..."
    
    # Stop any running processes
    echo "  - Stopping any running uvicorn processes..."
    pkill -f "python.*uvicorn" || true
    pkill -f "uvicorn src.api.routes:app" || true
    
    # Remove temporary files
    echo "  - Removing temporary files..."
    rm -f /tmp/debug_output.jpg
    
    # Clear model cache if desired
    read -p "Do you want to clear the model cache? This will require re-downloading models (y/N): " clear_cache
    if [[ "$clear_cache" == "y" || "$clear_cache" == "Y" ]]; then
        echo "  - Clearing model cache..."
        rm -rf "$HOME/.cache/image_stylist"
        rm -rf "$HOME/.cache/huggingface"
        echo "✅ Model cache cleared"
    fi
    
    echo "✅ Cleanup complete"
    exit 0
fi

# Test mode: run tests
if [[ "$MODE" == "test" ]]; then
    echo "🧪 Running tests..."
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        echo "✅ Activating virtual environment for tests..."
        source venv/bin/activate
    fi
    
    # Check and install test dependencies
    echo "Checking test dependencies..."
    python -c "
import importlib.util
missing = []
for pkg in ['pytest', 'pytest_cov', 'pytest_asyncio', 'httpx']:
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg.replace('_', '-'))
if missing:
    print(','.join(missing))
" > missing_deps.txt
    
    if [ -s missing_deps.txt ]; then
        echo "Installing missing test dependencies..."
        pip install $(cat missing_deps.txt)
    fi
    rm missing_deps.txt
    
    # Run the tests with coverage
    echo "Running tests with coverage..."
    python -m pytest tests/ -v --cov=src --cov-report=term-missing
    
    exit $?
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Create or activate virtual environment
if [ -d "venv" ]; then
    echo "✅ Activating existing virtual environment..."
    source venv/bin/activate
else
    echo "🔄 Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "🐍 Upgrading pip..."
    pip install --upgrade pip
    
    # Install specific PyTorch version FIRST (important for dependencies)
    PYTORCH_VERSION="2.5.1"
    echo "🔄 Installing PyTorch version ${PYTORCH_VERSION}..."
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "  (Platform: Apple Silicon or Intel Mac)"
        pip install torch==${PYTORCH_VERSION} torchvision # No specific index needed for recent macOS versions
    else
        echo "  (Platform: Linux/Other - Installing CPU version)"
        pip install torch==${PYTORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    # No need to install requirements here, will be done below
fi

# ALWAYS install/update dependencies from requirements.txt after activating/creating venv
echo "🔄 Ensuring dependencies from requirements.txt are installed..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found. Skipping dependency installation."
fi

# Ensure temporary directories exist
mkdir -p /tmp
if [ -f "/tmp/debug_output.jpg" ]; then
    chmod 644 /tmp/debug_output.jpg
fi

# Check and print library versions
echo "🔍 Checking library versions..."
python3 -c "
import sys
import platform
import torch
import diffusers
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
try:
    print(f'Diffusers: {diffusers.__version__}')
except Exception:
    print('Diffusers: Not found or error getting version')
print(f'Device: {platform.processor()}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'Apple Silicon MPS available: Yes')
else:
    print(f'Apple Silicon MPS available: No')
"

# Set environment variables for Apple Silicon performance
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "🔧 Setting up Apple Silicon environment variables..."
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

# Set environment variable for model cache location
export MODEL_CACHE_DIR="$HOME/.cache/image_stylist"
mkdir -p "$MODEL_CACHE_DIR"

# Set Python path to include the project root
export PYTHONPATH="$PWD:$PYTHONPATH"

# Pass FORCE_CPU flag as an environment variable
export FORCE_CPU_FLAG=$FORCE_CPU 

# Check for any previously crashed processes
# More robust killing of potential lingering processes
echo "🧹 Stopping potentially lingering processes..."
pkill -f "uvicorn src.api.routes:app" || true
sleep 1 # Give processes a moment to die

# Start the application
echo "🚀 Starting Image Style Transfer application..."
echo "📱 Open http://localhost:8081 in your web browser"
echo "ℹ️  Press Ctrl+C to stop the application"

# Run with proper Python path and environment (NO --reload for now)
echo "ℹ️ NOTE: Running without --reload for accurate startup check."
if [[ "$FORCE_CPU_FLAG" == "true" ]]; then
  echo " Bypassing MPS/GPU detection)"
fi
PYTHONPATH="$PWD" python -m uvicorn src.api.routes:app --host 0.0.0.0 --port 8081 --log-level info

# Check the exit code of uvicorn
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ ERROR: Application failed to start correctly (Exit Code: $EXIT_CODE). Please check logs above for details." >&2
    exit $EXIT_CODE
else
    # Note: This part might not be reached if uvicorn runs indefinitely until Ctrl+C
    # but it's good practice for script completion.
    echo "✅ Application finished normally." 
fi
