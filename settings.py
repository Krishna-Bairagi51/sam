# settings.py
import os # <--- Make sure os is imported

# --- Determine the Project Base Directory ---
# __file__ is the path to this settings.py file
# os.path.abspath gets the full absolute path
# os.path.dirname gets the directory containing the file
# BASE_DIR will be the absolute path to the 'Serverless' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# GroundingDINO paths - Use os.path.join to build absolute paths
GROUNDING_DINO_CONFIG_PATH = os.path.join(BASE_DIR, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(BASE_DIR, "weights/groundingdino_swint_ogc.pth")

# SAM model settings - Use os.path.join to build absolute paths
SAM_CHECKPOINT_PATH = os.path.join(BASE_DIR, "weights/sam_vit_h_4b8939.pth")
SAM_ENCODER_VERSION = "vit_h"
SAM_DEVICE = "cuda"

# Threshold settings
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# Object classes
CLASSES = ["packet"]

# Optional: Print paths during startup to verify
# print(f"BASE_DIR: {BASE_DIR}")
# print(f"GROUNDING_DINO_CONFIG_PATH: {GROUNDING_DINO_CONFIG_PATH}")
# print(f"GROUNDING_DINO_CHECKPOINT_PATH: {GROUNDING_DINO_CHECKPOINT_PATH}")
# print(f"SAM_CHECKPOINT_PATH: {SAM_CHECKPOINT_PATH}")
