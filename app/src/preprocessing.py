"""
=============================================================================
Data Preprocessing Module for Fake Job Detection
=============================================================================

This module handles all data preprocessing tasks required for training and
inference with the BERT model for fake job detection.

Main Components:
    1. DataPreprocessor - Full preprocessing pipeline for training
    2. TextPreprocessorForInference - Lightweight preprocessor for predictions

Key Features:
    - Text cleaning (remove HTML, URLs, special characters)
    - Tokenization using BERT tokenizer
    - SMOTE for handling class imbalance
    - Train/validation/test splitting

Author: ORFD Project Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard libraries
import re
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Optional

# Data processing
import pandas as pd
import numpy as np

# Machine learning utilities
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# BERT tokenizer
from transformers import BertTokenizer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import project configuration
from config.settings import MODEL_CONFIG, SMOTE_CONFIG, DATASET_CONFIG, DATA_DIR

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA PREPROCESSOR (FOR TRAINING)
# =============================================================================

class DataPreprocessor:
    """
    Preprocessor for preparing job posting data for BERT training.
    
    This class handles the complete data preprocessing pipeline:
        1. Loading dataset from CSV
        2. Cleaning text (removing HTML, URLs, etc.)
        3. Combining multiple text fields into single text
        4. Tokenizing text for BERT
        5. Applying SMOTE for class balance
        6. Splitting into train/validation/test sets
    
    Attributes:
        max_length: Maximum sequence length for BERT tokenization
        tokenizer: BERT tokenizer instance
    
    Example:
        >>> preprocessor = DataPreprocessor()
        >>> df = preprocessor.load_dataset("data/jobs.csv")
        >>> df = preprocessor.preprocess_dataset(df)
        >>> train_enc, val_enc, test_enc, train_y, val_y, test_y, *_ = preprocessor.prepare_splits(df)
    """
    
    def __init__(self, max_length: int = MODEL_CONFIG["max_length"]):
        """
        Initialize the DataPreprocessor.
        
        Args:
            max_length: Maximum token length for BERT (default: 256)
        """
        self.max_length = max_length
        
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
        
        logger.info(f"DataPreprocessor initialized with max_length={max_length}")
    
    def load_dataset(self, filepath: str = None) -> pd.DataFrame:
        """
        Load dataset from a CSV file.
        
        If no file is provided or the file doesn't exist, creates
        a sample dataset for testing purposes.
        
        Args:
            filepath: Path to the CSV file containing job postings
        
        Returns:
            Pandas DataFrame with job posting data
        """
        # Try to load from file if provided
        if filepath and Path(filepath).exists():
            dataframe = pd.read_csv(filepath)
            logger.info(f"Loaded {len(dataframe)} records from {filepath}")
            return dataframe
        
        # Create sample data if no file
        logger.info("Creating sample dataset...")
        return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create a sample dataset for testing.
        
        This is useful when no real data is available. Creates a mix
        of legitimate and fraudulent job posting examples.
        
        Returns:
            DataFrame with sample job postings
        """
        # Sample legitimate job posting
        legitimate_job = {
            "title": "Software Engineer",
            "company_profile": "Leading tech company with 500+ employees.",
            "description": "Skilled engineer for cutting-edge projects.",
            "requirements": "Bachelor's in CS, 3+ years exp.",
            "benefits": "Health insurance, 401k, flexible hours.",
            "fraudulent": 0
        }
        
        # Sample fraudulent job posting
        fraudulent_job = {
            "title": "EARN $5000/WEEK!!!",
            "company_profile": "",
            "description": "Make money fast! No experience! Send bank details!",
            "requirements": "None!",
            "benefits": "UNLIMITED EARNING!",
            "fraudulent": 1
        }
        
        # Create dataset with multiple copies
        legitimate_samples = [legitimate_job.copy() for _ in range(50)]
        fraudulent_samples = [fraudulent_job.copy() for _ in range(10)]
        
        # Combine into DataFrame
        all_samples = legitimate_samples + fraudulent_samples
        dataframe = pd.DataFrame(all_samples)
        
        # Add job IDs
        dataframe['job_id'] = range(len(dataframe))
        
        logger.info(f"Created sample dataset with {len(dataframe)} records")
        return dataframe
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize a text string.
        
        Removes:
            - HTML tags (e.g., <p>, <br>)
            - URLs (http://, www.)
            - Email addresses
            - Special characters (keeping only letters, numbers, basic punctuation)
            - Extra whitespace
        
        Args:
            text: Raw text string to clean
        
        Returns:
            Cleaned text string
        
        Example:
            >>> clean_text("<p>Visit http://site.com!</p>")
            "visit site.com"
        """
        # Handle missing values
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Keep only alphanumeric characters and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def combine_text(self, row: pd.Series) -> str:
        """
        Combine multiple job posting fields into a single text string.
        
        Combines: title, company_profile, description, requirements, benefits
        
        Args:
            row: A pandas Series representing one job posting
        
        Returns:
            Combined and cleaned text string
        """
        # List of fields to combine
        fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        
        # Get each field, clean it, and join with spaces
        combined_parts = []
        for field in fields:
            field_value = row.get(field, '')
            cleaned_value = self.clean_text(str(field_value))
            combined_parts.append(cleaned_value)
        
        return ' '.join(combined_parts)
    
    def preprocess_dataset(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess an entire dataset of job postings.
        
        Creates a 'combined_text' column with cleaned, combined text
        and filters out rows with very short text.
        
        Args:
            dataframe: Raw dataset with job posting columns
        
        Returns:
            Preprocessed DataFrame with 'combined_text' column
        """
        # Add combined text column
        dataframe['combined_text'] = dataframe.apply(self.combine_text, axis=1)
        
        # Filter out rows with very short text (less than 10 characters)
        original_length = len(dataframe)
        dataframe = dataframe[dataframe['combined_text'].str.len() > 10]
        dataframe = dataframe.reset_index(drop=True)
        
        filtered_count = original_length - len(dataframe)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} rows with text too short")
        
        return dataframe
    
    def tokenize_texts(self, texts: List[str]) -> dict:
        """
        Tokenize a list of texts using the BERT tokenizer.
        
        Args:
            texts: List of text strings to tokenize
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def apply_smote(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance the dataset.
        
        SMOTE (Synthetic Minority Over-sampling Technique) creates
        synthetic samples of the minority class to balance the dataset.
        
        Args:
            features: Feature array (X)
            labels: Label array (y)
        
        Returns:
            Tuple of (resampled_features, resampled_labels)
        """
        # Log original class distribution
        original_counts = np.bincount(labels)
        logger.info(f"Original class distribution: {original_counts}")
        
        # Determine number of neighbors (must be less than minority count)
        minority_count = sum(labels == 1)
        k_neighbors = min(SMOTE_CONFIG["k_neighbors"], minority_count - 1)
        
        # Create SMOTE sampler
        smote = SMOTE(
            sampling_strategy=SMOTE_CONFIG["sampling_strategy"],
            k_neighbors=k_neighbors,
            random_state=SMOTE_CONFIG["random_state"]
        )
        
        # Apply SMOTE
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        
        # Log new distribution
        new_counts = np.bincount(labels_resampled)
        logger.info(f"After SMOTE class distribution: {new_counts}")
        
        return features_resampled, labels_resampled
    
    def prepare_splits(
        self,
        dataframe: pd.DataFrame,
        apply_smote: bool = True
    ) -> Tuple:
        """
        Prepare train/validation/test splits from the dataset.
        
        Args:
            dataframe: Preprocessed DataFrame with 'combined_text' and 'fraudulent' columns
            apply_smote: Whether to apply SMOTE to training data
        
        Returns:
            Tuple containing:
                - train_encodings: Tokenized training texts
                - val_encodings: Tokenized validation texts
                - test_encodings: Tokenized test texts
                - train_labels: Training labels
                - val_labels: Validation labels
                - test_labels: Test labels
                - train_texts: Original training texts
                - val_texts: Original validation texts
                - test_texts: Original test texts
        """
        # Extract texts and labels
        texts = dataframe['combined_text'].tolist()
        labels = dataframe['fraudulent'].values
        
        # First split: separate test set
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=DATASET_CONFIG["test_size"],
            random_state=DATASET_CONFIG["random_state"],
            stratify=labels  # Maintain class proportions
        )
        
        # Second split: separate validation from training
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=DATASET_CONFIG["validation_size"],
            random_state=DATASET_CONFIG["random_state"],
            stratify=train_val_labels
        )
        
        logger.info(
            f"Data splits: Train={len(train_texts)}, "
            f"Val={len(val_texts)}, Test={len(test_texts)}"
        )
        
        # Tokenize all splits
        train_encodings = self.tokenize_texts(train_texts)
        val_encodings = self.tokenize_texts(val_texts)
        test_encodings = self.tokenize_texts(test_texts)
        
        # Convert labels to numpy arrays
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)
        
        return (
            train_encodings, val_encodings, test_encodings,
            train_labels, val_labels, test_labels,
            train_texts, val_texts, test_texts
        )


# =============================================================================
# TEXT PREPROCESSOR FOR INFERENCE
# =============================================================================

class TextPreprocessorForInference:
    """
    Lightweight text preprocessor for making predictions.
    
    This is a simpler version of DataPreprocessor, optimized for
    processing single texts during inference (prediction time).
    
    Attributes:
        tokenizer: BERT tokenizer instance
        max_length: Maximum sequence length
    
    Example:
        >>> preprocessor = TextPreprocessorForInference()
        >>> text = preprocessor.preprocess("Raw job posting text...")
        >>> tokens = preprocessor.tokenize(text)
    """
    
    def __init__(self, tokenizer_name: str = MODEL_CONFIG["model_name"]):
        """
        Initialize the inference preprocessor.
        
        Args:
            tokenizer_name: Name of the BERT tokenizer to use
        """
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = MODEL_CONFIG["max_length"]
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text for inference.
        
        Applies the same cleaning steps as DataPreprocessor.clean_text().
        
        Args:
            text: Raw text to preprocess
        
        Returns:
            Cleaned text string
        """
        # Handle missing values
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Keep only alphanumeric and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> dict:
        """
        Tokenize text for BERT input.
        
        Args:
            text: Preprocessed text to tokenize
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        # Preprocess first, then tokenize
        cleaned_text = self.preprocess(text)
        
        return self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    print("DataPreprocessor initialized successfully!")
    
    # Test text cleaning
    test_text = "<p>Visit http://example.com for more info!</p>"
    cleaned = preprocessor.clean_text(test_text)
    print(f"Cleaned text: '{cleaned}'")
