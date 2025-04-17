# syntax=docker/dockerfile:1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 1) Install system dependencies in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      wget \
      build-essential \
      ffmpeg libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 2) Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy in your entrypoint script and handler
COPY entrypoint.sh rp_handler.py ./

# 4) Make the entrypoint executable
RUN chmod +x entrypoint.sh

# 5) Use the script to auto‑detect CUDA_HOME and then exec your handler
ENTRYPOINT ["./entrypoint.sh"]
