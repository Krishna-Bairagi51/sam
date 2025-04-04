# rp_handler.py
import runpod
import base64
import io
from model_config import KuberaModel
from api_handler import detect_and_segment_object

# Load segmentation model(s) on startup
print("Loading Kubera segmentation models...")
MODEL_OBJ = KuberaModel()
try:
    MODEL_OBJ.load_model()
    print("Models loaded.")
except Exception as e:
    print(f"FATAL ERROR: Failed to load models: {e}")
    raise  # Reraise to prevent the worker from starting

def handler(event):
    print("Worker Start")
    input_data = event.get('input', {})
    image_data = input_data.get('image', None)  # Expecting a base64‑encoded image string

    if not image_data:
        return {"error": "No image data received."}

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_io = io.BytesIO(image_bytes)

        # Perform segmentation
        print("Running segmentation...")
        result = detect_and_segment_object(
            image_input=image_io,
            grounding_dino_model=MODEL_OBJ.GROUNDING_DINO_MODEL,
            sam_predictor=MODEL_OBJ.SAM_PREDICTOR
        )
        print("Segmentation complete.")
        return result
    except Exception as e:
        print("Error during processing:", str(e))
        return {"error": "Exception occurred", "details": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
