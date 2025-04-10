# rp_handler.py
import runpod
import base64
import io
import traceback  # For detailed error logging
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
    print(traceback.format_exc())  # Print the full traceback for loading errors
    raise  # Prevent the worker from starting if models cannot be loaded

def validate_event(event):
    """
    Validates that the incoming event (job payload) has the required fields.
    Expected structure:
        {
            "id": "<unique_job_identifier>",
            "input": { "images": [ ... ] }
        }
    
    Returns:
        tuple(bool, str): (True, "") if validation passes,
                           (False, error_message) otherwise.
    """
    if not isinstance(event, dict):
        return False, "Event payload must be a dictionary."
    if "id" not in event:
        return False, "Event payload is missing the 'id' field."
    if "input" not in event:
        return False, "Event payload is missing the 'input' field."
    return True, ""

def handler(event):
    print("Worker Start: Processing new event.")
    
    # Validate event structure
    is_valid, error_msg = validate_event(event)
    if not is_valid:
        print(f"Validation error: {error_msg}")
        return {"error": error_msg}
    
    input_data = event.get('input', {})
    # Expect a list of base64 images under the key "images"
    images_data_b64 = input_data.get('images', None)

    # --- Input Validation for the "images" key ---
    if images_data_b64 is None:
        err = "Input JSON must contain an 'images' key with a list of base64 strings."
        print(f"ERROR: {err}")
        return {"error": err}
    if not isinstance(images_data_b64, list):
        err = f"'images' key must contain a list of base64 strings. Received type: {type(images_data_b64)}"
        print(f"ERROR: {err}")
        return {"error": err}
    if len(images_data_b64) == 0:
        print("WARN: Received an empty list under 'images' key. Returning empty results.")
        return {"results": []}

    print(f"Received {len(images_data_b64)} image(s) to process.")
    batch_results = []  # To store results for each image

    # --- Process each image in the batch ---
    for idx, image_b64 in enumerate(images_data_b64):
        print(f"--- Processing image {idx+1}/{len(images_data_b64)} ---")
        if not isinstance(image_b64, str):
            err_detail = f"Item at index {idx} is not a string (expected base64). Skipping this image."
            print(f"ERROR: {err_detail}")
            batch_results.append({"error": f"Item at index {idx} is not a base64 string.", "details": None})
            continue  # Skip this image

        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_b64)
            image_io = io.BytesIO(image_bytes)

            # Perform segmentation for the current image.
            # The index 'idx' is passed to detect_and_segment_object for unique debug naming.
            segmented_objects_b64 = detect_and_segment_object(
                image_input=image_io,
                grounding_dino_model=MODEL_OBJ.GROUNDING_DINO_MODEL,
                sam_predictor=MODEL_OBJ.SAM_PREDICTOR,
                batch_index=idx  # Debug parameter to help with file naming, if needed
            )
            print(f"Image {idx+1}: Found {len(segmented_objects_b64)} segmented object(s).")
            batch_results.append({"segments": segmented_objects_b64})
        except base64.binascii.Error as b64_error:
            err_detail = f"Invalid base64 string for image at index {idx}: {str(b64_error)}"
            print(f"ERROR processing image {idx+1}: {err_detail}")
            batch_results.append({"error": f"Invalid base64 string for image at index {idx}.", "details": str(b64_error)})
        except Exception as e:
            err_detail = f"Exception processing image at index {idx}: {str(e)}"
            print(f"ERROR processing image {idx+1}: {err_detail}")
            print(traceback.format_exc())  # Print full traceback for debugging
            batch_results.append({"error": f"Exception processing image at index {idx}.", "details": str(e)})

    print(f"Finished processing batch. Returning {len(batch_results)} result entries.")
    # Return results structured per image
    # Example output: 
    # {"results": [{"segments": ["b64_obj1", "b64_obj2"]},
    #              {"error": "...", "details": "..."}, 
    #              {"segments": ["b64_obj3"]}]}
    return {"results": batch_results}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
