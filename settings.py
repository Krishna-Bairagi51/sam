# settings.py
import os # Keep os import if needed elsewhere, but not for BASE_DIR calculation for these paths

# --- Remove or comment out BASE_DIR calculation if only used for these paths ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --------------------------------------------

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "app.log"

# CORS settings
ALLOWED_ORIGINS = ["*"]
CORS_SETTINGS = {
    "allow_origins": ALLOWED_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Model settings

# GEMMA Model (No longer used)
GEMMA_MODEL = ""

# --- Use paths relative to the WORKDIR (/app) ---
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
# -----------------------------------------------

SAM_ENCODER_VERSION = "vit_h"
SAM_DEVICE = "cuda"

# Threshold settings
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# Object classes
CLASSES = ["packet"]

# Optional: Verify paths relative to expected CWD
# print(f"Expecting CWD: /app")
# print(f"Relative GROUNDING_DINO_CONFIG_PATH: {GROUNDING_DINO_CONFIG_PATH}")
# print(f"Relative GROUNDING_DINO_CHECKPOINT_PATH: {GROUNDING_DINO_CHECKPOINT_PATH}")
# print(f"Relative SAM_CHECKPOINT_PATH: {SAM_CHECKPOINT_PATH}")
