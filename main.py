from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model_config import KuberaModel
import io
from api_handler import detect_and_segment_object
app= FastAPI()
MODEL_OBJ = KuberaModel()
MODEL_OBJ.load_model()

@app.post('/segment')
async def segment_image(image:UploadFile=File(...)):
    import base64
    file_bytes = await image.read()
    image_io = io.BytesIO(file_bytes)

    sliced_image= detect_and_segment_object(image_input=image_io, grounding_dino_model=MODEL_OBJ.GROUNDING_DINO_MODEL, sam_predictor=MODEL_OBJ.SAM_PREDICTOR)
    print(sliced_image)
    return sliced_image