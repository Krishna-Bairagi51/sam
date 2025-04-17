# syntax=docker/dockerfile:1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app
COPY . /app

# Install OS packages (including git, ffmpeg, OpenCV deps) in one RUN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      ffmpeg \
      libsm6 \
      libxext6 \
      libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy and execute segmentation script with verbose logging
COPY segmentation.sh /app/segmentation.sh
RUN chmod +x /app/segmentation.sh && \
    bash -x /app/segmentation.sh

EXPOSE 8000
CMD ["python3", "-u", "rp_handler.py"]
