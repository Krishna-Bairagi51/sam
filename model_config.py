from fastapi import HTTPException
# from utilities.logger import LOGGER
from settings import (
    GROUNDING_DINO_CONFIG_PATH,
    GROUNDING_DINO_CHECKPOINT_PATH,
    SAM_ENCODER_VERSION,
    SAM_CHECKPOINT_PATH,
    SAM_DEVICE,
)
import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


class KuberaModel:
    def __init__(
        self,
        grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
        grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        sam_checkpoint_path=SAM_CHECKPOINT_PATH,
    ):
        self.grounding_dino_config_path = grounding_dino_config_path
        self.grounding_dino_checkpoint_path = grounding_dino_checkpoint_path
        self.sam_checkpoint_path = sam_checkpoint_path
        self.SAM_PREDICTOR = None
        self.GROUNDING_DINO_MODEL = None
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer from the pretrained directory."""
        try:
            print(f'loading models')
            # Loading the GROUNDING_DINO_MODEL
            self.GROUNDING_DINO_MODEL = Model(
                model_config_path=self.grounding_dino_config_path,
                model_checkpoint_path=self.grounding_dino_checkpoint_path,
            )
            print(f'loaded Grouding DINO', self.GROUNDING_DINO_MODEL)

            # Loading SAM_PREDICTOR
            SAM = sam_model_registry[SAM_ENCODER_VERSION](
                checkpoint=self.sam_checkpoint_path
            ).to(device=SAM_DEVICE)
            self.SAM_PREDICTOR = SamPredictor(SAM)
            print(f'loaded SAM')

                        
  
        except Exception as e:
            HTTPException(status_code=500, detail=f"Error loading the model: {e}")



# Initialize the model and vector database
# MODEL_OBJ = KuberaModel()


# Access loaded components as needed

# if MODEL_OBJ.GROUNDING_DINO_MODEL is not None:
#     LOGGER.info("Grounding DINO model is ready for Detection.")

# if MODEL_OBJ.SAM_PREDICTOR is not None:
#     LOGGER.info("SAM model is ready for sagmentation.")



