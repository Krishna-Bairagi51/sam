# api_handler.py
from settings import BOX_TRESHOLD, TEXT_TRESHOLD, CLASSES
from segment_anything import SamPredictor
import numpy as np
import cv2
from PIL import Image
import os, base64
import io # Import io for BytesIO

# --- Segmentation Function ---
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Uses SAM predictor to generate masks for given bounding boxes.

    Args:
        sam_predictor: The initialized SAM predictor instance.
        image: The input image as an RGB NumPy array.
        xyxy: A NumPy array of bounding boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        A NumPy array containing boolean masks for each bounding box.
    """
    sam_predictor.set_image(image) # Expects RGB image
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True # Get multiple masks for robustness
        )
        # Select the mask with the highest score
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks) # Returns a list of boolean masks

# --- Slicing Function ---
def slice_image_using_mask(original_image, mask, target_size=(1024, 1024)):
    """
    Extracts an object defined by a mask from an image, resizes it maintaining
    aspect ratio, and places it centered on a white background.

    Args:
        original_image: The source image (expected in RGB format).
        mask: The boolean or uint8 mask defining the object.
        target_size: The desired output image size (width, height).

    Returns:
        A NumPy array (RGB, uint8) of the sliced, resized, and centered object
        on a white background. Returns a blank white image if the mask is empty.
    """
    # Ensure mask is a NumPy array
    mask = np.asarray(mask)

    # Ensure mask is uint8 (0 or 255) for OpenCV functions
    if mask.dtype == bool:
        mask_uint8 = np.uint8(mask) * 255
    elif mask.dtype == np.uint8:
        # Ensure values are 0 or 255 if it's already uint8
        mask_uint8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    else:
        # Handle unexpected mask types if necessary, or raise error
        raise ValueError(f"Unsupported mask dtype: {mask.dtype}. Expected bool or uint8.")

    # Ensure the mask is contiguous in memory
    mask_uint8 = np.ascontiguousarray(mask_uint8)

    # Check dimensions before bitwise_and (This was the likely source of the original error)
    if original_image.shape[:2] != mask_uint8.shape[:2]:
        raise ValueError(f"Dimension mismatch: Original image shape {original_image.shape[:2]} != Mask shape {mask_uint8.shape[:2]}")

    # Extract the object using the mask (applies mask to the RGB image)
    try:
        original_image_contiguous = np.ascontiguousarray(original_image)
        object_only = cv2.bitwise_and(original_image_contiguous, original_image_contiguous, mask=mask_uint8)
    except cv2.error as e:
        print(f"ERROR in cv2.bitwise_and:")
        print(f"  original_image shape: {original_image.shape}, dtype: {original_image.dtype}")
        print(f"  mask_uint8 shape: {mask_uint8.shape}, dtype: {mask_uint8.dtype}")
        print(f"  OpenCV error: {e}")
        raise # Re-raise the exception

    # Get non-zero region from the mask to find bounding box
    coords = cv2.findNonZero(mask_uint8)
    if coords is None:
        print("WARN: Empty mask encountered in slice_image_using_mask. Returning blank image.")
        return np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the object using the mask's bounding rectangle
    cropped_object = object_only[y : y + h, x : x + w]
    cropped_mask_uint8 = mask_uint8[y : y + h, x : x + w]

    # --- Resize the cropped object to fit within target_size ---
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h) if w > 0 and h > 0 else 1 # Prevent division by zero
    new_w, new_h = int(w * scale), int(h * scale)

    # Ensure new dimensions are at least 1x1 if scale resulted in zero
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    resized_object = cv2.resize(cropped_object, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask_uint8 = cv2.resize(cropped_mask_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized_mask_uint8 = np.where(resized_mask_uint8 > 0, 255, 0).astype(np.uint8)

    # --- Create background canvas and place object ---
    result_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    mask_for_placement = resized_mask_uint8[:, :, np.newaxis] > 0

    result_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = np.where(
        mask_for_placement,
        resized_object,
        result_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w]
    )

    return result_image

# --- Main Detection and Segmentation Function ---
def detect_and_segment_object(image_input, grounding_dino_model, sam_predictor):
    """
    Detects objects using GroundingDINO, segments them using SAM,
    extracts each object, saves the base64 locally, and returns them
    as base64 encoded PNG images.

    Args:
        image_input: File-like object (e.g., io.BytesIO) containing the image data.
        grounding_dino_model: Initialized GroundingDINO model instance.
        sam_predictor: Initialized SAM predictor instance.

    Returns:
        A list of base64 encoded strings, each representing a segmented object image (PNG).
        Returns an empty list if no objects are detected.
    """
    print(f"DEBUG: Received image_input of type {type(image_input)}")

    # --- Image Loading and Preparation ---
    try:
        img_pil = Image.open(image_input).convert("RGB")
        image_np_rgb = np.array(img_pil)
    except Exception as e:
        print(f"ERROR: Failed to load or convert image: {e}")
        raise

    print(f"DEBUG: Loaded image_np_rgb shape: {image_np_rgb.shape}, dtype: {image_np_rgb.dtype}")

    # --- Object Detection ---
    try:
        detections = grounding_dino_model.predict_with_classes(
            image=image_np_rgb,
            classes=CLASSES,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )
    except Exception as e:
        print(f"ERROR: GroundingDINO prediction failed: {e}")
        raise

    # --- Handle No Detections ---
    if len(detections.xyxy) == 0:
        print("INFO: No objects detected matching the criteria.")
        return []

    print(f"DEBUG: Found {len(detections.xyxy)} initial detections.")

    # --- Object Segmentation ---
    try:
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=image_np_rgb,
            xyxy=detections.xyxy,
        )
        print(f"DEBUG: Generated {len(detections.mask)} masks.")
        if len(detections.mask) != len(detections.xyxy):
             print(f"WARN: Number of masks ({len(detections.mask)}) does not match number of detections ({len(detections.xyxy)}).")
    except Exception as e:
        print(f"ERROR: SAM segmentation failed: {e}")
        raise

    # --- Slicing, Encoding, and Saving ---
    sliced_images_b64 = []
    # Ensure debug directories exist
    os.makedirs("debug/img", exist_ok=True)
    os.makedirs("debug/b64", exist_ok=True) # <<< CREATE B64 DEBUG DIR

    for i in range(len(detections)):
        if i >= len(detections.mask):
            print(f"WARN: Skipping detection {i} due to missing mask.")
            continue

        mask = detections.mask[i]
        print(f"DEBUG Loop {i}: Processing detection. Mask shape: {mask.shape}, dtype: {mask.dtype}")

        try:
            # Slice image using the RGB image and the specific mask
            seg_img_rgb = slice_image_using_mask(image_np_rgb, mask)

            # Convert the resulting segmented image (RGB) to BGR format for OpenCV functions
            seg_img_bgr = cv2.cvtColor(seg_img_rgb, cv2.COLOR_RGB2BGR)
            print(f"DEBUG Loop {i}: Sliced seg_img_bgr shape: {seg_img_bgr.shape}, dtype: {seg_img_bgr.dtype}")

            # --- Encode to PNG ---
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            success, seg_buffer = cv2.imencode('.png', seg_img_bgr, encode_params)

            if not success:
                print(f"ERROR: Failed to encode image {i} to PNG format.")
                continue # Skip this image

            # --- Encode buffer to Base64 ---
            b64_string = base64.b64encode(seg_buffer).decode("utf-8")
            sliced_images_b64.append(b64_string) # Add to the list to be returned

            # --- Save Base64 string to local file (.b64 extension) --- # <<< NEW BLOCK
            b64_filename = f"debug/b64/slice_api_{i}.b64"
            try:
                with open(b64_filename, 'w', encoding='utf-8') as f_b64:
                    f_b64.write(b64_string)
                print(f"DEBUG: Saved base64 string to: {b64_filename}")
            except IOError as io_err:
                print(f"WARN: Failed to save base64 string to {b64_filename}: {io_err}")
            # --- End of saving base64 string --- #

            # --- Save debug image (optional PNG) ---
            debug_img_filename = f"debug/img/slice_api_{i}.png"
            save_success = cv2.imwrite(debug_img_filename, seg_img_bgr)
            if save_success:
                print(f"DEBUG: Saved debug image: {debug_img_filename}")
            else:
                print(f"WARN: Failed to save debug image: {debug_img_filename}")

        except ValueError as ve:
            print(f"ERROR Loop {i}: ValueError during slicing: {ve}")
            continue
        except cv2.error as cv_err:
             print(f"ERROR Loop {i}: OpenCV error during slicing or processing: {cv_err}")
             continue
        except Exception as e:
            print(f"ERROR Loop {i}: Unexpected error during slicing/encoding/saving: {e}")
            # Depending on severity, you might want to re-raise 'e' here
            continue


    print(f"DEBUG: Finished processing. Returning {len(sliced_images_b64)} base64 encoded images.")
    return sliced_images_b64
