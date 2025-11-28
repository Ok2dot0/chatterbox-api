# Chatterbox TTS Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir -e .

# Install additional dependencies for HTTP API
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy the API server
COPY api_server.py .

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Expose the API port
EXPOSE 8000

# Default command to run the API server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
