"""
Configuration file for Visual Product Similarity System
Intermediate Level Deep Learning Project
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== DATASET CONFIGURATION ====================
DATASET_NAME = "stanford_online_products"
DATASET_PATH = DATA_DIR / "stanford_online_products"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

# Image preprocessing
IMG_SIZE = 224
IMG_RESIZE_MODE = 'resize'  # 'resize', 'crop', or 'pad'
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# ==================== MODEL CONFIGURATION ====================
# Feature Extraction Model
FEATURE_EXTRACTOR_MODEL = "resnet50"  # Options: resnet50, efficientnet_b0
PRETRAINED_WEIGHTS = "imagenet"
EMBEDDING_DIM = 2048  # ResNet50 output dimension

# Model settings
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = "cuda"  # 'cuda' or 'cpu'
USE_GPU = True

# ==================== FAISS CONFIGURATION ====================
# Vector Index Settings
INDEX_TYPE = "IVF"  # Options: "FLAT", "IVF", "HNSW"
N_CLUSTERS = 100    # For IVF index
N_PROBE = 10        # For IVF search

# Similarity Search
K_RETRIEVAL = 10    # Top-K products to retrieve
SIMILARITY_METRIC = "cosine"  # 'cosine' or 'l2'

# ==================== TRAINING CONFIGURATION ====================
# Not used for this transfer learning approach, but kept for reference
EPOCHS = 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# ==================== EVALUATION CONFIGURATION ====================
# Metrics
EVAL_K_VALUES = [1, 5, 10, 20]  # For Precision@K, Recall@K
EVAL_METRIC = "cosine_similarity"

# Test Set Size
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# ==================== STREAMLIT APP CONFIGURATION ====================
# UI Settings
APP_TITLE = "🛍️ Visual Product Similarity System"
APP_DESCRIPTION = "Find visually similar products using Deep Learning"
MAX_UPLOAD_SIZE = 10  # MB

# Display Settings
RESULTS_PER_PAGE = 10
DISPLAY_SIMILARITY_SCORE = True
DISPLAY_RANK = True

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

# ==================== FEATURE EXTRACTION CACHE ====================
# Cache pre-extracted embeddings
CACHE_EMBEDDINGS = True
EMBEDDINGS_CACHE_PATH = DATA_DIR / "embeddings_cache"
EMBEDDINGS_CACHE_PATH.mkdir(exist_ok=True)

# ==================== INFERENCE SETTINGS ====================
# Production inference
INFERENCE_BATCH_SIZE = 64
INFERENCE_TIMEOUT = 30  # seconds
MAX_CONCURRENT_QUERIES = 5

# ==================== DATA AUGMENTATION ====================
# For fine-tuning (if needed)
USE_DATA_AUG = False
AUG_PROBABILITY = 0.5
AUG_ROTATION = 15
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2

# ==================== EXPORT SETTINGS ====================
# Model and index export
EXPORT_MODEL_FORMAT = "torchscript"  # 'torchscript', 'onnx'
EXPORT_INDEX_FORMAT = "faiss"
COMPRESSION_ENABLED = False

print("✅ Configuration loaded successfully!")
