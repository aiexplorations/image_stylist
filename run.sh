#!/bin/bash

# Unified run script for Image Style Transfer application
# This script handles setup, dependency installation, and launching the application

echo "====== Image Style Transfer ======"

# Parse command line options
MODE="run"
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
    --help)
      echo "Usage: ./run.sh [OPTIONS]"
      echo ""
      echo "OPTIONS:"
      echo "  --clean      Remove cache and temporary files"
      echo "  --test       Run tests"
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
    echo "  - Stopping any running processes..."
    pkill -f "python.*app_server.py" || true
    
    # Remove temporary files
    echo "  - Removing temporary files..."
    rm -f /tmp/debug_output.jpg
    
    # Clear model cache if desired
    read -p "Do you want to clear the model cache? This will require re-downloading models (y/N): " clear_cache
    if [[ "$clear_cache" == "y" || "$clear_cache" == "Y" ]]; then
        echo "  - Clearing model cache..."
        rm -rf ~/.cache/image_stylist
        rm -rf ~/.cache/huggingface/hub
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
    
    # Check if pytest is installed
    if ! python -c "import pytest" &>/dev/null; then
        echo "Installing pytest and test dependencies..."
        pip install pytest httpx
    fi
    
    # Create a test image for debug endpoints
    python -c "from PIL import Image; img = Image.new('RGB', (100, 100), color='red'); img.save('/tmp/debug_output.jpg')"
    
    # Run the tests
    python -m pytest test_app.py -v
    
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
    
    # Install PyTorch based on platform
    if [[ "$(uname)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            echo "🔄 Apple Silicon detected, installing PyTorch with MPS support..."
            pip install torch torchvision
        else
            echo "🔄 Intel Mac detected, installing PyTorch..."
            pip install torch torchvision
        fi
    else
        echo "🔄 Installing PyTorch for CPU..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    echo "🔄 Installing other dependencies..."
    pip install -r requirements.txt
fi

# Make the app server script executable
chmod +x app_server.py

# Ensure temporary directories exist
mkdir -p /tmp
if [ -f "/tmp/debug_output.jpg" ]; then
    chmod 644 /tmp/debug_output.jpg
fi

# Display hardware information
echo "🔍 Checking available hardware..."
python3 -c "
import torch
import platform
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
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

# Check for any previously crashed processes
pkill -f "python.*app_server.py" || true

# Start the application
echo "🚀 Starting Image Style Transfer application..."
echo "📱 Open http://localhost:8081 in your web browser"
echo "ℹ️  Press Ctrl+C to stop the application"
python app_server.py
