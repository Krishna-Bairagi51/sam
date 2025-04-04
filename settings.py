
import os
# settings.py

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "app.log"

# CORS settings
ALLOWED_ORIGINS = ["*"]  # Adjust as needed
CORS_SETTINGS = {
    "allow_origins": ALLOWED_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Model settings
HOME = os.getcwd()

# GEMMA Model (No longer used)
GEMMA_MODEL = ""

# GroundingDINO paths
GROUNDING_DINO_CONFIG_PATH = "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/weights/groundingdino_swint_ogc.pth"

# SAM model settings
SAM_CHECKPOINT_PATH = "/weights/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_DEVICE = "cuda"

# Threshold settings
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# Object classes
CLASSES = ["packet"]
