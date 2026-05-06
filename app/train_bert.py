"""
=============================================================================
BERT Training Script for Fake Job Detection
=============================================================================

This script trains a BERT model on the EMSCAD fake job postings dataset.
It includes features to improve model performance:
    - SMOTE for handling class imbalance
    - Class weights for better fraud detection
    - Configurable training parameters

Usage:
    Basic training:
        python train_bert.py
    
    Custom epochs:
        python train_bert.py --epochs 4 --batch-size 8
    
    Training without SMOTE:
        python train_bert.py --no-smote
    
    CPU-only training:
        python train_bert.py --no-gpu

Dataset:
    Uses EMSCAD dataset from Kaggle:
    https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Author: ORFD Project Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import logging
import argparse
from pathlib import Path
import copy

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Data processing
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# ML utilities
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

# Local imports
from src.model import JobFraudDetector, JobPostingDataset
from config.settings import MODEL_DIR, DATA_DIR, MODEL_CONFIG


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA DOWNLOAD FUNCTION
# =============================================================================

def download_dataset() -> str:
    """
    Download the EMSCAD dataset from Kaggle if not already present.
    
    The dataset contains ~17,880 job postings with labels indicating
    whether each posting is fraudulent (1) or legitimate (0).
    
    Returns:
        Path to the downloaded CSV file
    
    Raises:
        FileNotFoundError: If download fails or CSV not found
    """
    # Check if dataset already exists locally
    local_path = DATA_DIR / "fake_job_postings.csv"
    
    if local_path.exists():
        logger.info(f"Dataset already exists at {local_path}")
        return str(local_path)
    
    # Download from Kaggle using kagglehub
    logger.info("Downloading EMSCAD dataset from Kaggle...")
    
    import kagglehub
    import shutil
    
    # Download the dataset
    download_path = kagglehub.dataset_download(
        "shivamb/real-or-fake-fake-jobposting-prediction"
    )
    
    # Find and copy the CSV file
    for csv_file in Path(download_path).glob("*.csv"):
        shutil.copy(csv_file, local_path)
        logger.info(f"Dataset saved to {local_path}")
        return str(local_path)
    
    raise FileNotFoundError("CSV file not found in downloaded dataset")


# =============================================================================
# DATA PREPARATION FUNCTION
# =============================================================================

def prepare_data(csv_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load and prepare the dataset for training.
    
    Args:
        csv_path: Path to the CSV file
        sample_size: Optional limit on dataset size (useful for testing)
    
    Returns:
        Preprocessed DataFrame with 'text' column containing combined fields
    """
    # Load the dataset
    logger.info(f"Loading dataset from {csv_path}")
    dataframe = pd.read_csv(csv_path)
    
    # Log dataset statistics
    total_count = len(dataframe)
    fraud_count = dataframe['fraudulent'].sum()
    logger.info(f"Total samples: {total_count}, Fraudulent: {fraud_count}")
    
    # Sample if requested (useful for faster testing)
    if sample_size and len(dataframe) > sample_size:
        logger.info(f"Sampling {sample_size} records...")
        
        # Sample while maintaining rough class proportions
        legitimate_samples = dataframe[dataframe['fraudulent'] == 0].sample(
            n=int(sample_size * 0.85),
            random_state=42
        )
        fraudulent_samples = dataframe[dataframe['fraudulent'] == 1].head(
            int(sample_size * 0.15)
        )
        
        dataframe = pd.concat([legitimate_samples, fraudulent_samples])
        dataframe = dataframe.reset_index(drop=True)
    
    # Combine text fields into single column
    text_fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    dataframe['text'] = dataframe[text_fields].fillna('').agg(' '.join, axis=1)
    
    # Filter out very short texts
    dataframe = dataframe[dataframe['text'].str.len() > 10]
    dataframe = dataframe.reset_index(drop=True)
    
    logger.info(f"Prepared {len(dataframe)} samples for training")
    
    return dataframe


# =============================================================================
# TOKENIZATION FUNCTION
# =============================================================================

def tokenize(
    texts: pd.Series,
    labels: pd.Series,
    tokenizer: BertTokenizer
) -> tuple:
    """
    Tokenize texts for BERT input.
    
    Args:
        texts: Series of text strings to tokenize
        labels: Series of corresponding labels
        tokenizer: BERT tokenizer instance
    
    Returns:
        Tuple of (encodings dictionary, labels array)
    """
    # Convert texts to list for tokenizer
    text_list = texts.tolist()
    
    # Tokenize with padding and truncation
    # Use max_length from config for better coverage
    encodings = tokenizer(
        text_list,
        truncation=True,
        padding=True,
        max_length=MODEL_CONFIG["max_length"],
        return_tensors='pt'
    )
    
    # Convert labels to numpy array
    labels_array = np.array(labels)
    
    return encodings, labels_array


# =============================================================================
# SMOTE APPLICATION FUNCTION
# =============================================================================

def apply_smote(encodings: dict, labels: np.ndarray) -> tuple:
    """
    Apply SMOTETomek to balance the dataset.
    
    SMOTETomek combines SMOTE with Tomek links removal:
    1. SMOTE creates synthetic samples of the minority class (fraudulent)
    2. Tomek links removes noisy/borderline samples from both classes
    
    This achieves 97.24% accuracy per the base research paper - slightly
    better than SMOBD SMOTE (97.03%).
    
    Args:
        encodings: Dictionary with 'input_ids' and 'attention_mask' tensors
        labels: Array of labels
    
    Returns:
        Tuple of (resampled encodings, resampled labels)
    """
    # Convert tensors to numpy for SMOTE
    input_ids = encodings['input_ids'].numpy()
    attention_mask = encodings['attention_mask'].numpy()
    
    # Get sequence length for later reconstruction
    sequence_length = input_ids.shape[1]
    
    # Flatten and combine features for SMOTE
    # SMOTE needs a 2D array, so we flatten the tokens and masks together
    combined_features = np.hstack([
        input_ids.reshape(len(input_ids), -1),
        attention_mask.reshape(len(attention_mask), -1)
    ])
    
    # Apply SMOTETomek (SMOTE + Tomek links)
    # This gives better results than basic SMOTE as per the research paper
    # Paper result: BERT + SMOTE TomekLinks = 97.24% accuracy (vs SMOBD 97.03%)
    original_count = len(labels)
    
    smote_tomek = SMOTETomek(random_state=42)
    resampled_features, resampled_labels = smote_tomek.fit_resample(combined_features, labels)
    
    logger.info(f"Applied SMOTETomek: {original_count} samples -> {len(resampled_labels)} samples")
    
    # Reconstruct the encodings from resampled features
    # First half is input_ids, second half is attention_mask
    resampled_input_ids = resampled_features[:, :sequence_length]
    resampled_attention_mask = resampled_features[:, sequence_length:]
    
    # Clip values to valid ranges
    # Input IDs should be 0 to vocab_size (30522 for BERT)
    resampled_input_ids = np.clip(resampled_input_ids, 0, 30522).astype(np.int64)
    
    # Attention mask should be 0 or 1
    resampled_attention_mask = np.clip(resampled_attention_mask, 0, 1).astype(np.int64)
    
    # Convert back to tensors
    resampled_encodings = {
        'input_ids': torch.tensor(resampled_input_ids),
        'attention_mask': torch.tensor(resampled_attention_mask)
    }
    
    return resampled_encodings, resampled_labels


# =============================================================================
# CLASS WEIGHT CALCULATION FUNCTION
# =============================================================================

def calculate_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Gives higher weight to the minority class (fraudulent) so the model
    pays more attention to getting those predictions correct.
    
    Args:
        labels: Array of labels (0s and 1s)
        device: PyTorch device (CPU or GPU)
    
    Returns:
        Tensor of weights, one per class
    """
    total_samples = len(labels)
    
    # Count samples in each class
    legitimate_count = (labels == 0).sum()
    fraudulent_count = (labels == 1).sum()
    
    # Calculate weights (inverse of frequency)
    # Higher weight for rarer class
    weight_legitimate = total_samples / (2 * legitimate_count)
    weight_fraudulent = total_samples / (2 * fraudulent_count)
    
    logger.info(f"Class weights - Legitimate: {weight_legitimate:.2f}, Fraudulent: {weight_fraudulent:.2f}")
    
    # Create tensor and move to device
    weights = torch.tensor(
        [weight_legitimate, weight_fraudulent],
        dtype=torch.float32
    ).to(device)
    
    return weights


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

from sklearn.model_selection import StratifiedKFold

def train_model(
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 1e-5,  # Lower learning rate for more careful training
    sample_size: int = None,
    use_gpu: bool = True,
    use_smote: bool = True,
    use_weights: bool = True,
    fraud_weight_boost: float = 1.2,  # Boost fraud class weight for higher recall
    threshold: float = 0.3,  # For logging/consistency
    n_splits: int = 5,  # For k-fold cross-validation
    early_stopping_patience: int = 2  # For early stopping
) -> tuple:
    """
    Train the BERT model for fake job detection.
    
    This is the main training function that:
        1. Downloads/loads the dataset
        2. Prepares and splits the data
        3. Applies SMOTE if enabled
        4. Calculates class weights if enabled
        5. Trains the model
        6. Evaluates on test set
        7. Saves the best model
    
    Args:
        epochs: Number of training epochs (default: 4)
        batch_size: Samples per batch (default: 8)
        learning_rate: Learning rate for optimizer (default: 2e-5)
        sample_size: Limit dataset size (None = use all data)
        use_gpu: Whether to use GPU if available (default: True)
        use_smote: Whether to apply SMOTE balancing (default: True)
        use_weights: Whether to use class weights (default: True)
    
    Returns:
        Tuple of (trained detector, metrics dictionary)
    """
    # === DEVICE SETUP ===
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using GPU for training")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training")
        
        # Use smaller sample size for CPU training if not specified
        if not sample_size:
            sample_size = 2000
            logger.info(f"CPU mode: limiting to {sample_size} samples")
    
    # === DATA PREPARATION ===
    dataset_path = download_dataset()
    dataframe = prepare_data(dataset_path, sample_size)
    
    # === CROSS-VALIDATION SETUP ===
    X = dataframe['text'].values
    y = dataframe['fraudulent'].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG["model_name"])

    all_fold_metrics = []
    best_fold_val_loss = float('inf')
    best_fold_model = None
    best_fold_metrics = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"=== Fold {fold+1}/{n_splits} ===")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_encodings, train_labels = tokenize(pd.Series(X_train), pd.Series(y_train), tokenizer)
        val_encodings, val_labels = tokenize(pd.Series(X_val), pd.Series(y_val), tokenizer)

        if use_smote:
            train_encodings, train_labels = apply_smote(train_encodings, train_labels)

        class_weights = None
        if use_weights:
            class_weights = calculate_class_weights(train_labels, device)
            # Boost fraud class weight for higher recall
            class_weights[1] = class_weights[1] * fraud_weight_boost

        detector = JobFraudDetector(device=str(device))
        detector.initialize_model()

        # Early stopping logic
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            detector.train(
                train_encodings=train_encodings,
                train_labels=train_labels,
                val_encodings=val_encodings,
                val_labels=val_labels,
                epochs=1,  # One epoch at a time
                batch_size=batch_size,
                learning_rate=learning_rate,
                class_weights=class_weights
            )
            # Save checkpoint after each epoch
            checkpoint_dir = MODEL_DIR / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"fold{fold+1}_epoch{epoch+1}.pt"
            torch.save(detector.model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            val_loss, val_accuracy, _, _ = detector.evaluate(DataLoader(JobPostingDataset(val_encodings, val_labels), batch_size=batch_size))
            logger.info(f"Fold {fold+1} Epoch {epoch+1}: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model for this fold
                best_model_state = copy.deepcopy(detector.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1} for fold {fold+1}")
                    break

        # Load best model state for this fold
        detector.model.load_state_dict(best_model_state)
        # Evaluate on validation set
        val_loss, val_accuracy, val_preds, val_trues = detector.evaluate(DataLoader(JobPostingDataset(val_encodings, val_labels), batch_size=batch_size))
        metrics = detector.get_detailed_metrics(val_trues, val_preds)
        all_fold_metrics.append(metrics)
        # Track best fold for saving
        if val_loss < best_fold_val_loss:
            best_fold_val_loss = val_loss
            best_fold_model = detector
            best_fold_metrics = metrics

    # Save the best model from all folds
    model_path = MODEL_DIR / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    best_fold_model.save_model(model_path)
    logger.info(f"Best cross-validated model saved to {model_path}")
    logger.info(f"Best fold metrics: {best_fold_metrics}")
    # Average metrics across folds
    avg_metrics = {k: np.mean([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0]}
    logger.info(f"Average cross-validation metrics: {avg_metrics}")

    # === LOG FINAL METRICS TO LOG FILE ===
    from datetime import datetime
    import json
    from config.settings import LOGS_DIR
    log_file = LOGS_DIR / "training_log.json"

    # CLEAR PREVIOUS LOGS: Overwrite with empty dict before new training
    training_log = {}

    training_log['final_metrics'] = {
        'timestamp': datetime.now().isoformat(),
        'best_fold_metrics': best_fold_metrics,
        'average_metrics': avg_metrics,
        'training_recommendations': {
            'learning_rate': learning_rate,
            'fraud_weight_boost': fraud_weight_boost,
            'threshold': threshold,
            'note': 'Lower learning rate and higher fraud class weight boost recall. Threshold is set in model.py.'
        }
    }
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Final metrics logged to {log_file}")

    return best_fold_model, avg_metrics


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT model for fake job detection with cross-validation and early stopping")
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs (default: 4)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--sample-size', type=int, default=None, help='Limit dataset size (default: use all data)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage (use CPU only)')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE class balancing')
    parser.add_argument('--no-weights', action='store_true', help='Disable class weights')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--early-stopping-patience', type=int, default=2, help='Patience for early stopping (default: 2)')
    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        use_gpu=not args.no_gpu,
        use_smote=not args.no_smote,
        use_weights=not args.no_weights,
        n_splits=args.n_splits,
        early_stopping_patience=args.early_stopping_patience,
        fraud_weight_boost=1.2,  # Boost fraud class weight for recall
        learning_rate=1e-5,      # Lower learning rate for careful training
        threshold=0.3            # For logging/consistency
    )
