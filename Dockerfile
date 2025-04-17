# syntax=docker/dockerfile:1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary scripts and source files
COPY segmentation.sh rp_handler.py ./

# Run model setup script
RUN chmod +x segmentation.sh && ./segmentation.sh

# Default command
CMD ["python3", "-u", "rp_handler.py"]
