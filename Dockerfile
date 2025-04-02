# Use Python as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8081 \
    PYTHONPATH=/app \
    MODEL_CACHE_DIR=/app/.cache/image_stylist \
    FORCE_CPU_FLAG=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch for CPU
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create cache directory
RUN mkdir -p /app/.cache/image_stylist

# Copy application files
COPY src /app/src
COPY style_prompt_guide.md /app/style_prompt_guide.md

# Start the application
CMD ["python", "-m", "uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8081"]

EXPOSE 8081
