"""
=============================================================================
Application Configuration Settings
=============================================================================

This module contains all configuration settings for the fake job detection
application. All settings are centralized here for easy modification.

Sections:
    1. Directory Paths - Where to find/store files
    2. Model Configuration - BERT model parameters
    3. Training Configuration - SMOTE and training settings
    4. Server Configuration - Flask and Streamlit settings
    5. Logging Configuration - Log format and level
    6. Label Mapping - Prediction label definitions

To modify settings:
    - Edit values directly in this file
    - Or set environment variables (see .env.example)

Author: ORFD Project Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
# This allows overriding settings without modifying code
load_dotenv()


# =============================================================================
# DIRECTORY PATHS
# =============================================================================

# Base directory is the 'app' folder (parent of this config folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory for storing datasets
DATA_DIR = BASE_DIR / "data"

# Directory for storing trained models
MODEL_DIR = BASE_DIR / "models"

# Directory for log files
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
# This ensures the application can always find these folders
for directory_path in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    directory_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# These settings control the BERT model architecture and training behavior

MODEL_CONFIG = {
    # Name of the pre-trained BERT model to use
    # Options: "bert-base-uncased", "bert-base-cased", "bert-large-uncased",
    #          "roberta-base", "microsoft/deberta-v3-base"
    # "uncased" means text is lowercased before processing
    # HIGHER ACCURACY OPTIONS (uncomment to use):
    # "model_name": "roberta-base",           # +1-3% accuracy
    # "model_name": "microsoft/deberta-v3-base",  # +3-5% accuracy (best)
    "model_name": "bert-base-uncased",
    
    # Maximum number of tokens (words/subwords) per input
    # Longer texts are truncated, shorter texts are padded
    # BERT maximum is 512 - INCREASE for better accuracy on long job posts
    "max_length": 384,  # Increased from 256 for better coverage
    
    # Number of samples to process at once during training
    # Larger = faster training but more memory usage
    # Reduce if you get out-of-memory errors
    "batch_size": 16,
    
    # Learning rate for the optimizer
    # Controls how much to adjust weights during training
    # 2e-5 (0.00002) is recommended for BERT fine-tuning
    "learning_rate": 2e-5,
    
    # Number of complete passes through the training data
    # More epochs = better training but risk of overfitting
    # INCREASED for better convergence
    "epochs": 5,  # Increased from 3
    
    # Number of output classes
    # 2 for binary classification (legitimate vs fraudulent)
    "num_labels": 2,
    
    # Dropout rate for regularization
    # Randomly zeros this fraction of neurons during training
    # REDUCED - 0.3 is often too aggressive for BERT
    "dropout_rate": 0.2,  # Reduced from 0.3
    
    # =========================================================================
    # ADVANCED SETTINGS FOR HIGHER ACCURACY (NEW)
    # =========================================================================
    
    # Warmup ratio - gradually increase learning rate at start
    # Helps prevent early divergence
    "warmup_ratio": 0.1,
    
    # Weight decay for L2 regularization
    "weight_decay": 0.01,
    
    # Gradient accumulation steps - effectively increases batch size
    # Use when GPU memory is limited
    "gradient_accumulation_steps": 2,
    
    # Early stopping patience - stop if no improvement for N epochs
    "early_stopping_patience": 3,
    
    # Label smoothing - prevents overconfidence
    "label_smoothing": 0.1,
}


# =============================================================================
# SMOTE CONFIGURATION
# =============================================================================
# SMOTETomek (SMOTE + Tomek Links) settings
# Combines oversampling with noise removal for better accuracy
# Paper result: BERT + SMOTETomek = 97.24% accuracy (vs basic SMOTE ~96%)

SMOTE_CONFIG = {
    # How to balance classes
    # "auto" = make minority class equal to majority class
    # Can also be a float like 0.5 = minority will be 50% of majority
    "sampling_strategy": "auto",
    
    # Number of nearest neighbors to use when creating synthetic samples
    # Must be less than the number of minority class samples
    "k_neighbors": 5,
    
    # Random seed for reproducibility
    # Using same seed gives same results every time
    "random_state": 42,
}


# =============================================================================
# FLASK SERVER CONFIGURATION
# =============================================================================
# Settings for the Flask backend API server

FLASK_CONFIG = {
    # Host address
    # "0.0.0.0" = accept connections from any IP
    # "localhost" or "127.0.0.1" = only accept local connections
    "host": os.getenv("FLASK_HOST", "0.0.0.0"),
    
    # Port number
    # Default is 5000, change if that port is in use
    "port": int(os.getenv("FLASK_PORT", 5000)),
    
    # Debug mode
    # True = auto-reload on code changes, detailed error messages
    # Should be False in production for security
    "debug": os.getenv("FLASK_DEBUG", "False").lower() == "true",
}


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================
# Settings for the Streamlit frontend

STREAMLIT_CONFIG = {
    # URL of the backend API
    # Update this if backend runs on different host/port
    "api_url": os.getenv("API_URL", "http://localhost:5000"),
}


# =============================================================================
# LIME CONFIGURATION
# =============================================================================
# LIME (Local Interpretable Model-agnostic Explanations) settings
# Used for explaining individual predictions

LIME_CONFIG = {
    # Number of top features (words) to show in explanations
    "num_features": 10,
    
    # Number of samples to generate for building local model
    # More samples = more accurate but slower explanations
    "num_samples": 500,
}


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Settings for splitting the dataset into train/validation/test sets

DATASET_CONFIG = {
    # Fraction of data to use for testing (0.2 = 20%)
    "test_size": 0.2,
    
    # Fraction of remaining data to use for validation (0.1 = 10%)
    # After test split, this gives roughly 70% train, 10% val, 20% test
    "validation_size": 0.1,
    
    # Random seed for reproducibility
    "random_state": 42,
}


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Settings for application logging

LOGGING_CONFIG = {
    # Minimum log level to display
    # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    "level": os.getenv("LOG_LEVEL", "INFO"),
    
    # Format for log messages
    # Includes: timestamp, logger name, level, message
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}


# =============================================================================
# LABEL MAPPING
# =============================================================================
# Maps numeric predictions to human-readable labels

# Converts model output (0 or 1) to label string
LABEL_MAP = {
    0: "Legitimate",   # Model predicts 0 -> job is legitimate
    1: "Fraudulent"    # Model predicts 1 -> job is fraudulent
}

# Reverse mapping: converts label string back to number
# Useful when processing user input
REVERSE_LABEL_MAP = {
    "Legitimate": 0,
    "Fraudulent": 1
}
