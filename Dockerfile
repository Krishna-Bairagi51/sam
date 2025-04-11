# Use an official Python image as a base
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
# It's often better to copy only necessary files first (requirements, scripts)
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
        build-essential && \ # Added build-essential for compiling C++/CUDA
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make the script executable and run it to install models and compile extensions
RUN chmod +x ./segmentation.sh && ./segmentation.sh

# Copy the rest of your application code
COPY . /app

# Expose the port (Optional for RunPod serverless, but good practice)
# EXPOSE 8000

# Command to run the server
CMD ["python3", "-u", "rp_handler.py"]
