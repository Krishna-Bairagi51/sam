# Use an official Python image as a base
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy necessary files first
COPY requirements.txt ./
COPY segmentation.sh ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies including build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        # Ensure build tools for C++/CUDA compilation are present
        build-essential && \ # Ensure backslash is THE LAST character on this line
    # Clean up apt caches
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make the script executable and run it to install models/extensions
RUN chmod +x ./segmentation.sh && ./segmentation.sh

# Copy the rest of your application code
COPY . /app

# Expose the port (Optional for RunPod serverless)
# EXPOSE 8000

# Command to run the server
CMD ["python3", "-u", "rp_handler.py"]
