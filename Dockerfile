# syntax=docker/dockerfile:1

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# 1. Set working directory
WORKDIR /app

# 2. Copy application code
COPY . /app

# 3. Install system dependencies (Git, ffmpeg, X libs) in one layer,
#    using --no-install-recommends to avoid extra packages,
#    and clean up apt caches to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \                       # for cloning repos or version info :contentReference[oaicite:0]{index=0}
      ffmpeg \                    # for video/image processing :contentReference[oaicite:1]{index=1}
      libsm6 libxext6 libgl1 &&   # X11 libs for OpenCV GUI :contentReference[oaicite:2]{index=2}
    rm -rf /var/lib/apt/lists/*

# 4. Ensure segmentation.sh is executable and run it with bash -x
#    for verbose debugging (prints each command before execution) :contentReference[oaicite:3]{index=3}
COPY segmentation.sh /app/segmentation.sh
RUN chmod +x /app/segmentation.sh && \
    bash -x /app/segmentation.sh

# 5. Expose the application port
EXPOSE 8000

# 6. Default command
CMD ["python3", "-u", "rp_handler.py"]
