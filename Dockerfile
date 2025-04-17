# syntax=docker/dockerfile:1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git gcc g++ ffmpeg libsm6 libxext6 libgl1 && \
    ln -s /usr/local/cuda/bin/nvcc /usr/bin/nvcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy segmentation and source code
COPY segmentation.sh rp_handler.py ./

# Run SAM + Grounding DINO install
RUN chmod +x segmentation.sh && ./segmentation.sh

CMD ["python3", "-u", "rp_handler.py"]
