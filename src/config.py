# config.py
# All project settings and configurations in one place.

import os

# --- DIRECTORIES ---
# Base directory of the src folder
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Main project root directory (one level up from src)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images", "Dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Output directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# --- FILE PATHS ---
MODEL_PATH = os.path.join(MODEL_DIR, "crop_stress_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# --- MODEL PARAMETERS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001