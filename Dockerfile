FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1  -y
COPY segmentation.sh /app/segmentation.sh
RUN chmod +x /app/segmentation.sh && /app/segmentation.sh
EXPOSE 8000
CMD ["python3", "-u", "rp_handler.py"]  
