# Use an official Python image as a base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1  -y
# Copy and run the installation script for SAM and Grounding DINO
COPY segmentation.sh /app/segmentation.sh
RUN chmod +x /app/segmentation.sh && /app/segmentation.sh

# Expose the port for uvicorn
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
