# # rp_handler.py
# import runpod
# import base64
# import io
# from model_config import KuberaModel
# from api_handler import detect_and_segment_object

# # Load segmentation model(s) on startup
# print("Loading Kubera segmentation models...")
# MODEL_OBJ = KuberaModel()
# try:
#     MODEL_OBJ.load_model()
#     print("Models loaded.")
# except Exception as e:
#     print(f"FATAL ERROR: Failed to load models: {e}")
#     raise  # Reraise to prevent the worker from starting

# def handler(event):
#     print("Worker Start")
#     input_data = event.get('input', {})
#     image_data = input_data.get('image', None)  # Expecting a base64‑encoded image string

#     if not image_data:
#         return {"error": "No image data received."}

#     try:
#         # Decode base64 image
#         image_bytes = base64.b64decode(image_data)
#         image_io = io.BytesIO(image_bytes)

#         # Perform segmentation
#         print("Running segmentation...")
#         result = detect_and_segment_object(
#             image_input=image_io,
#             grounding_dino_model=MODEL_OBJ.GROUNDING_DINO_MODEL,
#             sam_predictor=MODEL_OBJ.SAM_PREDICTOR
#         )
#         print("Segmentation complete.")
#         return result
#     except Exception as e:
#         print("Error during processing:", str(e))
#         return {"error": "Exception occurred", "details": str(e)}

# if __name__ == "__main__":
#     runpod.serverless.start({"handler": handler})



# rp_handler.py
import runpod
import base64
import io
import traceback # Import traceback for detailed error logging
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
    print(traceback.format_exc()) # Print full traceback for loading errors
    raise  # Reraise to prevent the worker from starting

def handler(event):
    print("Worker Start: Processing new event.")
    input_data = event.get('input', {})
    # --- Expect a list of base64 images under the key "images" ---
    images_data_b64 = input_data.get('images', None)

    # --- Input Validation ---
    if not images_data_b64:
        print("ERROR: No 'images' key found in input.")
        return {"error": "Input JSON must contain an 'images' key with a list of base64 strings."}
    if not isinstance(images_data_b64, list):
        print(f"ERROR: 'images' key does not contain a list. Type received: {type(images_data_b64)}")
        return {"error": "'images' key must contain a list of base64 strings."}
    if not images_data_b64: # Check if list is empty
         print("WARN: Received an empty list under 'images' key.")
         return {"results": []} # Return empty results for an empty input list

    print(f"Received {len(images_data_b64)} image(s) to process.")
    batch_results = [] # To store results for each image [[results_img1], [results_img2], ...]

    # --- Process Each Image in the Batch ---
    for idx, image_b64 in enumerate(images_data_b64):
        print(f"--- Processing image {idx+1}/{len(images_data_b64)} ---")
        if not isinstance(image_b64, str):
            print(f"ERROR: Item at index {idx} is not a string (expected base64). Skipping.")
            batch_results.append({"error": f"Item at index {idx} is not a base64 string.", "details": None})
            continue # Skip to the next image

        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_b64)
            image_io = io.BytesIO(image_bytes)

            # Perform segmentation for the current image
            # Pass the index 'idx' to detect_and_segment_object for unique debug filenames
            segmented_objects_b64 = detect_and_segment_object(
                image_input=image_io,
                grounding_dino_model=MODEL_OBJ.GROUNDING_DINO_MODEL,
                sam_predictor=MODEL_OBJ.SAM_PREDICTOR,
                batch_index=idx # Pass the index for debug naming
            )
            print(f"Image {idx+1}: Found {len(segmented_objects_b64)} segmented objects.")
            batch_results.append({"segments": segmented_objects_b64}) # Append list of segments for this image

        except base64.binascii.Error as b64_error:
            print(f"ERROR processing image {idx+1}: Invalid base64 string. {b64_error}")
            batch_results.append({"error": f"Invalid base64 string for image at index {idx}.", "details": str(b64_error)})
        except Exception as e:
            print(f"ERROR processing image {idx+1}: {str(e)}")
            print(traceback.format_exc()) # Print full traceback for debugging
            batch_results.append({"error": f"Exception processing image at index {idx}.", "details": str(e)})

    print(f"Finished processing batch. Returning {len(batch_results)} result entries.")
    # --- Return results structured per image ---
    # Example output: {"results": [{"segments": ["b64_obj1", "b64_obj2"]}, {"error": "...", "details": "..."}, {"segments": ["b64_obj3"]}]}
    return {"results": batch_results}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
