from settings import BOX_TRESHOLD, TEXT_TRESHOLD, CLASSES
from segment_anything import SamPredictor
import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

def slice_image_using_mask(original_image, mask, target_size=(1024, 1024)):
        """
        Center-align and resize the segmented object using only the mask to fit within target dimensions.

        Args:
        - original_image (np.array): Original RGB image.
        - mask (np.array): Segmentation mask.
        - target_size (tuple): Desired (width, height) for the output image.

        Returns:
        - result_image (np.array): Processed image with the object centered and resized.
        """
        if mask.dtype == bool:
            mask = np.uint8(mask) * 255
        # # cv2.imwrite('mask.png', mask)
        # # Extract the object using the mask (set background to transparent)
        # object_only = cv2.bitwise_and(original_image, original_image, mask=mask)
        # # cv2.imwrite('object_only.png', object_only)
        # # Get non-zero region from the mask
        # coords = cv2.findNonZero(mask)
        # x, y, w, h = cv2.boundingRect(coords)  # Bounding box around the mask

        # # Crop the object using the mask's bounding rectangle
        # cropped_object = object_only[y : y + h, x : x + w]
        # cropped_mask = mask[y : y + h, x : x + w]
        # cv2.imwrite('cropped_object.png', cropped_object)
        # cv2.imwrite('cropped_mask.png', cropped_mask)

        # # Resize the cropped object to fit within target_size while preserving aspect ratio
        # target_w, target_h = target_size
        # scale_w = target_w / w
        # scale_h = target_h / h
        # scale = min(scale_w, scale_h)  # Choose the smallest scale to maintain aspect ratio

        # new_w, new_h = int(w * scale), int(h * scale)
        # resized_object = cv2.resize(cropped_object, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # # Create a white background canvas of target_size
        # result_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

        # # Center the resized object on the white background
        # x_offset = (target_w - new_w) // 2
        # y_offset = (target_h - new_h) // 2

        # # Add only the object on the white background
        # result_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = np.where(
        #     resized_mask[:, :, None] > 0, resized_object, result_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w]
        # )

        return mask

def detect_and_segment_object(image_input, grounding_dino_model, sam_predictor):
        import os, base64
        print(f"DEBUG: Received image_input of type {type(image_input)}")
        image_np = np.array(Image.open(image_input).convert("RGB"))
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        detections = grounding_dino_model.predict_with_classes(
            image=image_np,
            classes=CLASSES,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )
        if len(detections.xyxy) > 0:
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=image_np,
                xyxy=detections.xyxy,
            )
        sliced_images = []
        for index, detection in enumerate(detections):
            seg_img=slice_image_using_mask(image_np, detection[1])
            encode_params= [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            _, seg_buffer= cv2.imencode('.png', seg_img, encode_params)
            b64= base64.b64encode(seg_buffer).decode("utf-8")
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
            sliced_images.append(
                b64
            )
            os.makedirs("debug/img", exist_ok=True)
            cv2.imwrite(f"debug/img/slice_api_{index}.png", seg_img)

        return sliced_images