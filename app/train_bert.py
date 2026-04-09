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

def train_model(
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    sample_size: int = None,
    use_gpu: bool = True,
    use_smote: bool = True,
    use_weights: bool = True
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
    
    # === DATA SPLITTING ===
    # Split into train (80%) and temp (20%)
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.2,
        stratify=dataframe['fraudulent'],
        random_state=42
    )
    
    # Split temp into validation (50%) and test (50%)
    # This gives us: 80% train, 10% validation, 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['fraudulent'],
        random_state=42
    )
    
    logger.info(
        f"Data split - Train: {len(train_df)}, "
        f"Validation: {len(val_df)}, Test: {len(test_df)}"
    )
    
    # === TOKENIZATION ===
    tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
    
    train_encodings, train_labels = tokenize(
        train_df['text'], train_df['fraudulent'], tokenizer
    )
    val_encodings, val_labels = tokenize(
        val_df['text'], val_df['fraudulent'], tokenizer
    )
    test_encodings, test_labels = tokenize(
        test_df['text'], test_df['fraudulent'], tokenizer
    )
    
    # === APPLY SMOTE (if enabled) ===
    if use_smote:
        train_encodings, train_labels = apply_smote(train_encodings, train_labels)
    
    # === CALCULATE CLASS WEIGHTS (if enabled) ===
    class_weights = None
    if use_weights:
        class_weights = calculate_class_weights(train_labels, device)
    
    # === MODEL INITIALIZATION ===
    detector = JobFraudDetector(device=str(device))
    detector.initialize_model()
    
    # === TRAINING ===
    logger.info(
        f"Starting training: epochs={epochs}, batch_size={batch_size}, "
        f"smote={use_smote}, class_weights={use_weights}"
    )
    
    detector.train(
        train_encodings=train_encodings,
        train_labels=train_labels,
        val_encodings=val_encodings,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # === EVALUATION ===
    logger.info("Evaluating on test set...")
    
    # Create test DataLoader
    test_dataset = JobPostingDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Evaluate
    test_loss, test_accuracy, predictions, true_labels = detector.evaluate(test_loader)
    
    # Get detailed metrics
    metrics = detector.get_detailed_metrics(true_labels, predictions)
    
    # Log results
    logger.info("=" * 60)
    logger.info("FINAL RESULTS:")
    logger.info(f"  Accuracy:          {metrics['accuracy']:.2%}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
    logger.info(f"  Precision:         {metrics['precision']:.2%}")
    logger.info(f"  Recall:            {metrics['recall']:.2%}")
    logger.info(f"  F1 Score:          {metrics['f1_score']:.2%}")
    logger.info("=" * 60)
    
    # === SAVE MODEL ===
    model_path = MODEL_DIR / "best_model.pt"
    detector.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return detector, metrics


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train BERT model for fake job detection"
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of training epochs (default: 4)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Limit dataset size (default: use all data)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage (use CPU only)'
    )
    
    parser.add_argument(
        '--no-smote',
        action='store_true',
        help='Disable SMOTE class balancing'
    )
    
    parser.add_argument(
        '--no-weights',
        action='store_true',
        help='Disable class weights'
    )
    
    args = parser.parse_args()
    
    # Run training with parsed arguments
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        use_gpu=not args.no_gpu,
        use_smote=not args.no_smote,
        use_weights=not args.no_weights
    )
