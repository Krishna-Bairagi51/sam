# Use an official Python image as a base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1  -y
# Copy and run the installation script for SAM and Grounding DINO
COPY segmentation.sh /app/segmentation.sh
RUN chmod +x /app/segmentation.sh && /app/segmentation.sh

# Expose the port for uvicorn
# EXPOSE 8000

# Command to run the server
CMD ["python3", "-u", "rp_handler.py"]
