FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# Set the working directory
WORKDIR /app

# Copy necessary files first

COPY requirements.txt ./
COPY segmentation.sh ./

RUN apt-get update && \
        libsm6 \
        libxext6 \
        libgl1 \
        # Ensure build tools for C++/CUDA compilation are present
        build-essential && \
    # Clean up apt caches
    apt-get clean && \
    # Remove apt lists to reduce image size
    rm -rf /var/lib/apt/lists/*

# Make the script executable and run it to install models/extensions
RUN chmod +x ./segmentation.sh && ./segmentation.sh

# Copy the rest of your application code
COPY . /app

# Expose the port (Optional for RunPod serverless)
# EXPOSE 8000

# Command to run the server
