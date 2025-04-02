# Use Python as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8081

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch for CPU
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY ollama_styler.html /var/www/html/index.html
COPY default.conf /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Configure nginx
RUN rm -f /etc/nginx/sites-enabled/default

# Create start script
RUN echo '#!/bin/bash\nnginx\npython3 app.py' > /app/start.sh && chmod +x /app/start.sh

# Start both nginx and Python app
CMD ["/app/start.sh"]

EXPOSE 8081
