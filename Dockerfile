# syntax=docker/dockerfile:1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies including Git and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git build-essential ffmpeg libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy segmentation script, handler, and entrypoint
COPY segmentation.sh rp_handler.py entrypoint.sh ./

# Build SAM and GroundingDINO extensions
RUN chmod +x segmentation.sh && ./segmentation.sh

# Entrypoint to auto-detect CUDA_HOME and launch handler
ENTRYPOINT ["./entrypoint.sh"]
