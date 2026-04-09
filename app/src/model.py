"""
=============================================================================
BERT-based Model for Fake Job Detection
=============================================================================

This module contains the core machine learning components for detecting
fraudulent job postings using BERT (Bidirectional Encoder Representations 
from Transformers).

Main Components:
    1. JobPostingDataset - PyTorch Dataset for handling job posting data
    2. BertClassifier - Neural network model combining BERT with classification layers
    3. JobFraudDetector - High-level class for training and making predictions

Author: ORFD Project Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

# PyTorch - Deep learning framework
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader

# Transformers - For BERT model
from transformers import BertModel, BertTokenizer

# Scikit-learn - For evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

# Standard libraries
import numpy as np
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import project configuration
from config.settings import MODEL_CONFIG, MODEL_DIR, LABEL_MAP, LOGS_DIR

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET CLASS
# =============================================================================

class JobPostingDataset(Dataset):
    """
    PyTorch Dataset class for job postings.
    
    This class wraps tokenized job posting text and labels so they can be
    efficiently loaded in batches during training.
    
    Attributes:
        encodings (dict): Tokenized text data with 'input_ids' and 'attention_mask'
        labels (np.ndarray): Array of labels (0=legitimate, 1=fraudulent)
    
    Example:
        >>> encodings = tokenizer(texts, return_tensors='pt', padding=True)
        >>> labels = np.array([0, 1, 0, 1])
        >>> dataset = JobPostingDataset(encodings, labels)
        >>> len(dataset)
        4
    """
    
    def __init__(self, encodings: dict, labels: np.ndarray):
        """
        Initialize the dataset with encodings and labels.
        
        Args:
            encodings: Dictionary containing tokenized text (input_ids, attention_mask)
            labels: Numpy array of integer labels
        """
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        # Create a dictionary with the encoded values for this sample
        item = {}
        for key, value in self.encodings.items():
            item[key] = value[idx].clone()
        
        # Add the label as a tensor
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


# =============================================================================
# BERT CLASSIFIER MODEL
# =============================================================================

class BertClassifier(nn.Module):
    """
    BERT-based binary classifier for fake job detection.
    
    Architecture:
        1. BERT Base Model (768 hidden dimensions)
        2. Dropout Layer (prevents overfitting)
        3. Fully Connected Layer: 768 -> 256
        4. ReLU Activation
        5. Dropout Layer
        6. Fully Connected Layer: 256 -> 64
        7. ReLU Activation
        8. Dropout Layer
        9. Output Layer: 64 -> 2 (legitimate/fraudulent)
    
    Attributes:
        bert: Pre-trained BERT model
        dropout: Dropout layer for regularization
        classifier: Sequential neural network for classification
        num_labels: Number of output classes (2 for binary classification)
    """
    
    def __init__(
        self,
        model_name: str = MODEL_CONFIG["model_name"],
        num_labels: int = MODEL_CONFIG["num_labels"],
        dropout: float = MODEL_CONFIG["dropout_rate"]
    ):
        """
        Initialize the BERT classifier.
        
        Args:
            model_name: Name of the pre-trained BERT model (default: 'bert-base-uncased')
            num_labels: Number of output classes (default: 2)
            dropout: Dropout probability (default: 0.3)
        """
        super().__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
        # Get the size of BERT's output (768 for bert-base)
        bert_hidden_size = self.bert.config.hidden_size
        
        # Classification head: reduces BERT output to final prediction
        self.classifier = nn.Sequential(
            # First layer: 768 -> 256
            nn.Linear(bert_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second layer: 256 -> 64
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer: 64 -> 2 (legitimate or fraudulent)
            nn.Linear(64, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        class_weights: torch.Tensor = None
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs from tokenizer (batch_size, sequence_length)
            attention_mask: Mask to ignore padding tokens (batch_size, sequence_length)
            token_type_ids: Segment IDs (optional, not used for single sentences)
            labels: Ground truth labels for computing loss (optional)
            class_weights: Weights for handling class imbalance (optional)
        
        Returns:
            Dictionary containing:
                - 'logits': Raw model outputs before softmax
                - 'probabilities': Softmax probabilities for each class
                - 'loss': Cross-entropy loss (only if labels provided)
        """
        # Pass input through BERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the pooled output (CLS token representation)
        pooled_output = bert_output.pooler_output
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Pass through classification layers
        logits = self.classifier(pooled_output)
        
        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)
        
        # Prepare result dictionary
        result = {
            'logits': logits,
            'probabilities': probabilities
        }
        
        # Compute loss if labels are provided (during training)
        if labels is not None:
            # Use weighted loss if class weights provided (helps with imbalanced data)
            if class_weights is not None:
                loss_function = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_function = nn.CrossEntropyLoss()
            
            result['loss'] = loss_function(logits, labels)
        
        return result


# =============================================================================
# JOB FRAUD DETECTOR (MAIN CLASS)
# =============================================================================

class JobFraudDetector:
    """
    Main class for training and using the job fraud detection model.
    
    This class provides a high-level interface for:
        - Training the BERT model on job posting data
        - Making predictions on new job postings
        - Evaluating model performance
        - Saving and loading trained models
    
    Attributes:
        device: CPU or GPU device for computation
        model: The BERT classifier model
        tokenizer: BERT tokenizer for converting text to tokens
        training_history: Dictionary tracking loss and accuracy during training
    
    Example:
        >>> detector = JobFraudDetector()
        >>> detector.load_model()  # Load pre-trained model
        >>> result = detector.predict("Software Engineer at Google...")
        >>> print(result['label'])  # 'Legitimate' or 'Fraudulent'
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Job Fraud Detector.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detects if not specified.
        """
        # Determine which device to use (GPU if available, else CPU)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model as None (loaded later)
        self.model = None
        
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
        
        # Dictionary to store training metrics over epochs
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        logger.info(f"JobFraudDetector initialized. Using device: {self.device}")
    
    def initialize_model(self):
        """
        Create a new BERT classifier model and move it to the appropriate device.
        Call this before training a new model.
        """
        self.model = BertClassifier()
        self.model = self.model.to(self.device)
        logger.info("New BERT classifier model initialized")
    
    def train(
        self,
        train_encodings: dict,
        train_labels: np.ndarray,
        val_encodings: dict,
        val_labels: np.ndarray,
        epochs: int = MODEL_CONFIG["epochs"],
        batch_size: int = MODEL_CONFIG["batch_size"],
        learning_rate: float = MODEL_CONFIG["learning_rate"],
        class_weights: torch.Tensor = None
    ) -> dict:
        """
        Train the model on the provided data.
        
        Args:
            train_encodings: Tokenized training texts
            train_labels: Training labels (0 or 1)
            val_encodings: Tokenized validation texts
            val_labels: Validation labels
            epochs: Number of training epochs (default: 3)
            batch_size: Samples per batch (default: 16)
            learning_rate: Learning rate for optimizer (default: 2e-5)
            class_weights: Weights for handling class imbalance (optional)
        
        Returns:
            Dictionary containing training history (losses and accuracies)
        """
        # Initialize model if not already done
        if not self.model:
            self.initialize_model()
        
        # Create DataLoaders for batching
        train_dataset = JobPostingDataset(train_encodings, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = JobPostingDataset(val_encodings, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer (AdamW is recommended for BERT fine-tuning)
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01  # L2 regularization
        )
        
        # Learning rate scheduler for warmup
        total_steps = len(train_loader) * epochs
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=total_steps // 10
        )
        
        # Track best validation accuracy for saving best model
        best_validation_accuracy = 0
        
        # Training loop
        for epoch in range(epochs):
            # ===== TRAINING PHASE =====
            self.model.train()  # Set model to training mode
            total_training_loss = 0
            
            # Progress bar for training batches
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for batch in progress_bar:
                # Zero gradients from previous step
                optimizer.zero_grad()
                
                # Move batch data to device (GPU/CPU)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    class_weights=class_weights
                )
                
                # Backward pass (compute gradients)
                loss = outputs['loss']
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update model weights
                optimizer.step()
                scheduler.step()
                
                # Accumulate loss
                total_training_loss += loss.item()
            
            # ===== VALIDATION PHASE =====
            val_loss, val_accuracy, _, _ = self.evaluate(val_loader)
            
            # Calculate average training loss for this epoch
            avg_train_loss = total_training_loss / len(train_loader)
            
            # Store metrics in history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"Val Accuracy = {val_accuracy:.4f}"
            )
            
            # Save model if it's the best so far
            is_best = val_accuracy > best_validation_accuracy
            if is_best:
                best_validation_accuracy = val_accuracy
                self.save_model(MODEL_DIR / "best_model.pt")
                logger.info(f"New best model saved! Accuracy: {val_accuracy:.4f}")
            
            # ===== SAVE TRAINING LOG TO FILE =====
            self._save_training_log(
                epoch=epoch + 1,
                epochs=epochs,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                is_best_model=is_best,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
        
        return self.training_history
    
    def _save_training_log(
        self,
        epoch: int,
        epochs: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        is_best_model: bool,
        learning_rate: float,
        batch_size: int
    ):
        """
        Save training metrics to a readable JSON log file.
        
        Creates/updates a JSON file at app/logs/training_log.json with
        metrics for each epoch. This file is human-readable unlike the
        binary .pt model checkpoint.
        
        Args:
            epoch: Current epoch number (1-indexed)
            epochs: Total number of epochs
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            val_accuracy: Validation accuracy for this epoch
            is_best_model: Whether this epoch produced the best model
            learning_rate: Learning rate used
            batch_size: Batch size used
        """
        log_file = LOGS_DIR / "training_log.json"
        
        # Load existing log or create new
        if log_file.exists():
            with open(log_file, 'r') as f:
                training_log = json.load(f)
        else:
            training_log = {
                'training_started': datetime.now().isoformat(),
                'model_config': {
                    'model_name': MODEL_CONFIG['model_name'],
                    'max_length': MODEL_CONFIG['max_length'],
                    'dropout_rate': MODEL_CONFIG['dropout_rate']
                },
                'hyperparameters': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'total_epochs': epochs
                },
                'epochs': [],
                'best_accuracy': 0.0,
                'best_epoch': 0
            }
        
        # Add this epoch's metrics
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'val_accuracy': round(val_accuracy, 6),
            'val_accuracy_percent': round(val_accuracy * 100, 2),
            'is_best_model': is_best_model
        }
        
        # Update or append epoch data
        existing_epochs = [e['epoch'] for e in training_log['epochs']]
        if epoch in existing_epochs:
            idx = existing_epochs.index(epoch)
            training_log['epochs'][idx] = epoch_data
        else:
            training_log['epochs'].append(epoch_data)
        
        # Update best metrics
        if is_best_model:
            training_log['best_accuracy'] = round(val_accuracy * 100, 2)
            training_log['best_epoch'] = epoch
        
        # Update last modified time
        training_log['last_updated'] = datetime.now().isoformat()
        
        # Save to file
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        logger.info(f"Training log saved to {log_file}")
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader containing the evaluation data
        
        Returns:
            Tuple containing:
                - Average loss
                - Accuracy score
                - Predictions array
                - True labels array
        """
        self.model.eval()  # Set model to evaluation mode
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Accumulate loss
                total_loss += outputs['loss'].item()
                
                # Get predictions (class with highest logit)
                predictions = torch.argmax(outputs['logits'], dim=1)#picks class with highest logit score as prediction
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
        
        # Calculate metrics
        average_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return average_loss, accuracy, np.array(all_predictions), np.array(all_labels)
    
    def get_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate detailed evaluation metrics.
        
        Args:
            y_true: Array of true labels
            y_pred: Array of predicted labels
        
        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy
                - balanced_accuracy: Accuracy accounting for class imbalance
                - precision: Precision score
                - recall: Recall score (fraud detection rate)
                - f1_score: Harmonic mean of precision and recall
                - confusion_matrix: 2x2 confusion matrix
                - classification_report: Detailed per-class metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(
                y_true, y_pred,
                target_names=['Legitimate', 'Fraudulent'],
                output_dict=True
            )
        }
        return metrics
    
    def detect_fraud_signals(self, text: str) -> Dict:
        """
        Detect rule-based fraud signals in job posting text.
        
        These rules catch common scam patterns that the ML model might miss:
            - Upfront fees (asking applicant to pay money)
            - Personal email domains (gmail, yahoo, hotmail, outlook)
            - Money handling requests (wire transfers, gift cards)
            - Excessive urgency language
            - Unrealistic salary promises
        
        Args:
            text: The job posting text to analyze
        
        Returns:
            Dictionary with:
                - signals: List of detected fraud signals
                - score: Fraud score boost (0.0 to 1.0)
        """
        import re
        text_lower = text.lower()
        signals = []
        score = 0.0
        
        # === UPFRONT FEES (Very strong indicator) ===
        fee_patterns = [
            r'(pay|send|transfer|fee).{0,30}(\$\d+|\d+\s*dollars)',
            r'(processing|registration|training|application)\s*fee',
            r'(purchase|buy).{0,20}(software|equipment|kit|supplies)',
            r'small\s*(startup|registration|one.?time)\s*fee',
            # Catch Indian Rupee amounts with purchase/fee context
            r'(purchase|buy|pay|cost).{0,40}(₹|rs\.?|inr)\s*[\d,]+',
            r'(₹|rs\.?|inr)\s*[\d,]+.{0,40}(kit|fee|cost|purchase|required)',
            # Onboarding/training kit purchase (sophisticated scam pattern)
            r'(onboarding|training|starter)\s*(kit|package|materials).{0,30}(₹|rs\.?|\$|cost|purchase)',
            r'required\s*to\s*(purchase|buy|pay)',
        ]
        for pattern in fee_patterns:
            if re.search(pattern, text_lower):
                signals.append("Requests upfront payment/fees")
                score += 0.4
                break
        
        # === DELAYED REIMBURSEMENT SCAM (Strong indicator) ===
        # "Cost will be reimbursed after X months" is a classic scam tactic
        reimbursement_patterns = [
            r'(reimbursed?|refunded?).{0,30}(after|following).{0,20}(month|week|day|employment)',
            r'(cost|fee|amount).{0,30}(reimbursed?|refunded?).{0,20}(after|upon)',
            r'(deducted|recovered).{0,20}(from|through).{0,20}(salary|paycheck)',
        ]
        for pattern in reimbursement_patterns:
            if re.search(pattern, text_lower):
                signals.append("Delayed reimbursement promise (scam tactic)")
                score += 0.35
                break
        
        # === PERSONAL EMAIL DOMAINS (Strong indicator) ===
        personal_email_pattern = r'@(gmail|yahoo|hotmail|outlook|aol|mail)\.(com|net|org)'
        if re.search(personal_email_pattern, text_lower):
            signals.append("Uses personal email domain")
            score += 0.25
        
        # === SUSPICIOUS DOMAIN PATTERNS ===
        # Domains with hyphens trying to look legitimate (e.g., nexacore-tech.co)
        suspicious_domain_pattern = r'@[\w]+-[\w]+\.(co|io|xyz|info|biz)\b'
        if re.search(suspicious_domain_pattern, text_lower):
            signals.append("Suspicious email domain pattern")
            score += 0.15
        
        # === GIFT CARDS (Very strong indicator - classic scam) ===
        if re.search(r'gift\s*card', text_lower):
            signals.append("Mentions gift cards")
            score += 0.5
        
        # === WIRE/TRANSFER MONEY ===
        if re.search(r'(wire|transfer|receive).{0,20}(money|funds|payment)', text_lower):
            signals.append("Involves money transfers")
            score += 0.3
        
        # === IMMEDIATE CHECK/PAYMENT ===
        if re.search(r'(send|receive|mail).{0,20}(check|cheque)', text_lower):
            signals.append("Involves receiving checks")
            score += 0.35
        
        # === PROPRIETARY SOFTWARE/EQUIPMENT PURCHASE ===
        # Scams often require purchasing "proprietary" tools
        proprietary_patterns = [
            r'proprietary\s*(software|tools?|system|platform).{0,30}(purchase|buy|cost|access)',
            r'(software|tools?)\s*access.{0,20}(included|part of).{0,20}(kit|package|fee)',
            r'(includes?|contains?).{0,20}(software\s*access|training\s*materials).{0,20}(cost|fee|₹|\$)',
        ]
        for pattern in proprietary_patterns:
            if re.search(pattern, text_lower):
                signals.append("Requires proprietary software/equipment purchase")
                score += 0.3
                break
        
        # === NO EXPERIENCE/INTERVIEW NEEDED ===
        no_exp_patterns = [
            r'no\s*(experience|interview|background\s*check)\s*(needed|required)',
            r'(experience|interview)\s*not\s*(needed|required)',
        ]
        for pattern in no_exp_patterns:
            if re.search(pattern, text_lower):
                signals.append("No experience/interview required")
                score += 0.15
                break
        
        # === URGENCY TACTICS ===
        urgency_patterns = [
            r'(act|apply|start)\s*now',
            r'limited\s*(time|positions|spots)',
            r'(urgent|immediate)(ly)?\s*(hiring|opening|need)',
        ]
        urgency_count = sum(1 for p in urgency_patterns if re.search(p, text_lower))
        if urgency_count >= 2:
            signals.append("Uses urgency tactics")
            score += 0.15
        
        # === GUARANTEED INCOME ===
        if re.search(r'guaranteed\s*(income|salary|pay|earnings|\$)', text_lower):
            signals.append("Promises guaranteed income")
            score += 0.2
        
        # === UNREALISTIC PAY ===
        # Looking for very high weekly/daily pay claims
        if re.search(r'\$\s*[2-9]\d{3,}\s*(/|\s*per\s*)(week|wk)', text_lower):
            signals.append("Unrealistic weekly pay")
            score += 0.2
        
        # === EXCESSIVE CAPS/EXCLAMATION ===
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        exclaim_count = text.count('!')
        if caps_ratio > 0.3 or exclaim_count > 5:
            signals.append("Excessive caps/exclamation marks")
            score += 0.1
        
        # === WORK FROM HOME + HIGH PAY ===
        if (re.search(r'work\s*from\s*home', text_lower) and 
            re.search(r'\$\s*[3-9]\d{2,}.*/(hour|hr)', text_lower)):
            signals.append("Work from home with high hourly pay")
            score += 0.15
        
        # === WHATSAPP/TELEGRAM CONTACT (Medium indicator) ===
        # Legitimate companies rarely use messaging apps for recruitment
        messaging_patterns = [
            r'(whatsapp|telegram|signal)\s*[:\-@]?\s*\+?\d',
            r'contact.{0,20}(via|through|on)\s*(whatsapp|telegram)',
            r'@\w+.*telegram',
            r'\+\d{1,3}[\-\s]?\d{3,}.*whatsapp',
            r'via\s+whatsapp',
            r'whatsapp\s*(at|:)',
        ]
        for pattern in messaging_patterns:
            if re.search(pattern, text_lower):
                signals.append("Uses messaging apps (WhatsApp/Telegram) for hiring")
                score += 0.2
                break
        
        # === SECURITY/REFUNDABLE DEPOSIT (Strong indicator) ===
        deposit_patterns = [
            r'(security|refundable|registration)\s*deposit',
            r'deposit.{0,20}(refund|return)',
            r'\$\d+.{0,30}(refundable|security)',
        ]
        for pattern in deposit_patterns:
            if re.search(pattern, text_lower):
                signals.append("Requires deposit (potential scam)")
                score += 0.35
                break
        
        # === PACKAGE RESHIPPING/LOGISTICS SCAM ===
        reshipping_patterns = [
            r'(receive|accept)\s*(packages?|shipments?).{0,20}(home|address)',
            r'repackage.{0,20}(items?|products?|goods?)',
            r'forward.{0,20}(packages?|shipments?).{0,20}(international|overseas)',
            r'(inspect|check).{0,15}(and|&).{0,15}(repackage|forward)',
        ]
        for pattern in reshipping_patterns:
            if re.search(pattern, text_lower):
                signals.append("Package reshipping scheme (common scam)")
                score += 0.4
                break
        
        # === ASSESSMENT/CERTIFICATION FEE ===
        assessment_patterns = [
            r'(certified|certification|assessment)\s*(program|course|test).{0,30}\$\d+',
            r'\$\d+.{0,30}(assessment|certification|test|program)',
            r'(must|required to).{0,20}(complete|pass).{0,20}(paid|fee)',
            r'(candidates?|applicants?).{0,20}(must|required).{0,30}(complete|pay).{0,30}\$\d+',
            r'assessment.{0,10}program.{0,10}\(\$\d+\)',
        ]
        for pattern in assessment_patterns:
            if re.search(pattern, text_lower):
                signals.append("Paid assessment/certification required")
                score += 0.35
                break
        
        # === REIMBURSEMENT AFTER DAYS/WEEKS (Broader pattern) ===
        delayed_reimburse_patterns = [
            r'reimburs.{0,20}(within|after)\s*\d+\s*(day|week|month)',
            r'(full|complete)\s*reimburs.{0,20}\d+\s*(day|week)',
        ]
        for pattern in delayed_reimburse_patterns:
            if re.search(pattern, text_lower) and "Delayed reimbursement promise" not in signals:
                signals.append("Delayed reimbursement promise (scam tactic)")
                score += 0.35
                break
        
        # === CRYPTO/TRADING DEPOSIT SCAM ===
        crypto_patterns = [
            r'(unlock|activate).{0,20}(trading|account).{0,20}(deposit|payment)',
            r'(security|starter)\s*deposit.{0,20}(crypto|trading|portfolio)',
            r'practice.{0,15}(live\s*)?trading.{0,20}(company|our)\s*funds',
        ]
        for pattern in crypto_patterns:
            if re.search(pattern, text_lower):
                signals.append("Trading/crypto deposit scam")
                score += 0.4
                break
        
        # Cap the score at 0.95
        score = min(score, 0.95)
        
        return {
            'signals': signals,
            'score': score
        }
    
    def extract_posting_details(self, text: str) -> Dict:
        """
        Extract and analyze specific details from job posting text.
        
        This method extracts structured information like emails, salary,
        contact methods, company mentions, and provides detailed advisories.
        
        Args:
            text: The job posting text to analyze
        
        Returns:
            Dictionary with extracted details and advisory information
        """
        import re
        text_lower = text.lower()
        details = {
            'extracted_info': {},
            'detailed_advisories': []
        }
        
        # ================================================================
        # 1. EMAIL EXTRACTION AND ANALYSIS
        # ================================================================
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text_lower)
        
        if emails:
            details['extracted_info']['emails'] = emails
            
            for email in emails:
                domain = email.split('@')[1] if '@' in email else ''
                
                # Check for personal email domains
                personal_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
                                   'aol.com', 'mail.com', 'protonmail.com', 'icloud.com']
                
                if any(pd in domain for pd in personal_domains):
                    details['detailed_advisories'].append({
                        'category': 'Contact Email',
                        'finding': f'Personal email detected: {email}',
                        'risk_level': 'Medium-High',
                        'advisory': f'''We found a personal email address ({email}) in this job posting. 
                        
**Why this matters:** Legitimate companies typically use corporate email addresses (like hr@companyname.com) for recruitment. Personal email services (Gmail, Yahoo, etc.) are commonly used in scam postings because they're free and anonymous.

**What you should do:**
1. Search for the company's official website
2. Look for their official "Careers" or "Contact" page
3. Verify if this email matches their official contact information
4. If you can't find the company online, this is a significant red flag''',
                        'icon': '📧'
                    })
                else:
                    # Corporate email - still advise verification
                    company_domain = domain.split('.')[0] if domain else 'unknown'
                    details['detailed_advisories'].append({
                        'category': 'Contact Email',
                        'finding': f'Corporate email found: {email}',
                        'risk_level': 'Low',
                        'advisory': f'''This posting uses a corporate email domain ({domain}).

**This is a positive sign**, but you should still verify:
1. Visit {domain} directly (type it in your browser, don't click links)
2. Check if they have an active careers page listing this position
3. Look up the company on LinkedIn to verify they're hiring
4. Confirm the email format matches their other job postings''',
                        'icon': '✉️'
                    })
        
        # ================================================================
        # 2. SALARY/PAY EXTRACTION AND ANALYSIS
        # ================================================================
        # Various salary patterns
        salary_patterns = [
            (r'\$\s*(\d{1,3}(?:,\d{3})*)\s*(?:/|\s*per\s*)(hour|hr)', 'hourly'),
            (r'\$\s*(\d{1,3}(?:,\d{3})*)\s*(?:/|\s*per\s*)(week|wk)', 'weekly'),
            (r'\$\s*(\d{1,3}(?:,\d{3})*)\s*(?:/|\s*per\s*)(month|mo)', 'monthly'),
            (r'\$\s*(\d{1,3}(?:,\d{3})*)\s*(?:/|\s*per\s*)(year|yr|annual)', 'annual'),
            (r'\$\s*(\d{1,3}(?:,\d{3})*)\s*[-–]\s*\$?\s*(\d{1,3}(?:,\d{3})*)', 'range'),
            (r'(\d{1,3}(?:,\d{3})*)\s*(?:dollars?)\s*(?:/|\s*per\s*)(week|hour|month)', 'text'),
        ]
        
        salary_mentions = []
        for pattern, pay_type in salary_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                salary_mentions.append((matches, pay_type))
        
        if salary_mentions:
            details['extracted_info']['salary_mentions'] = str(salary_mentions)
            
            # Check for unrealistic amounts
            unrealistic = False
            for matches, pay_type in salary_mentions:
                for match in matches:
                    amount = int(match[0].replace(',', '')) if isinstance(match, tuple) else int(match.replace(',', ''))
                    
                    if pay_type == 'hourly' and amount > 100:
                        unrealistic = True
                    elif pay_type == 'weekly' and amount > 3000:
                        unrealistic = True
                    elif pay_type == 'monthly' and amount > 15000 and 'experience' not in text_lower:
                        unrealistic = True
            
            if unrealistic:
                details['detailed_advisories'].append({
                    'category': 'Salary Claims',
                    'finding': 'Unusually high pay mentioned',
                    'risk_level': 'High',
                    'advisory': '''The salary mentioned in this posting appears unusually high.

**Why this is concerning:** Scam job postings often promise extremely high pay to attract victims. Legitimate jobs rarely offer $5,000+/week or $100+/hour for entry-level or "no experience needed" positions.

**How to verify:**
1. Search typical salaries for this role on Glassdoor, Indeed, or LinkedIn
2. Compare with similar positions in your area
3. Ask yourself: "Does this make sense for the work described?"
4. Remember: If it sounds too good to be true, it usually is

**Industry averages for reference:**
- Data Entry: $15-25/hour
- Administrative Assistant: $18-30/hour  
- Customer Service: $15-25/hour
- Professional roles typically require experience for high pay''',
                    'icon': '💰'
                })
            else:
                details['detailed_advisories'].append({
                    'category': 'Salary Information',
                    'finding': 'Salary/pay rate mentioned',
                    'risk_level': 'Info',
                    'advisory': '''A salary or pay rate was mentioned in this posting.

**To verify this is reasonable:**
1. Compare with similar positions on job sites like Glassdoor, Indeed, or LinkedIn Salary
2. Consider the location - pay varies significantly by region
3. Consider the experience level required
4. Check if the pay matches the responsibilities described''',
                    'icon': '📊'
                })
        
        # ================================================================
        # 3. CONTACT METHOD ANALYSIS
        # ================================================================
        # Phone numbers
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        
        if phones:
            details['extracted_info']['phone_numbers'] = phones
        
        # WhatsApp/Telegram (common in scams)
        if re.search(r'whatsapp|telegram|signal', text_lower):
            details['extracted_info']['messaging_apps'] = True
            details['detailed_advisories'].append({
                'category': 'Contact Method',
                'finding': 'Messaging app contact (WhatsApp/Telegram) mentioned',
                'risk_level': 'Medium-High',
                'advisory': '''This posting asks you to contact via WhatsApp, Telegram, or similar messaging apps.

**Why this raises concerns:** While some legitimate companies use messaging apps, scammers strongly prefer them because:
- Messages are encrypted and hard to trace
- Accounts are easy to create anonymously
- They can quickly abandon accounts if reported

**What to do:**
1. Never share personal information via messaging apps before verifying the employer
2. Look for official company communication channels first
3. Legitimate recruiters typically use company email or established platforms (LinkedIn, Indeed)
4. Be extra cautious if this is the ONLY contact method provided''',
                'icon': '💬'
            })
        
        # ================================================================
        # 4. COMPANY INFORMATION ANALYSIS
        # ================================================================
        # Check for company indicators
        company_patterns = [
            r'(?:company|employer|organization|firm|agency):\s*([^\n\.,]+)',
            r'(?:at|for|with)\s+([A-Z][A-Za-z\s&]+(?:Inc|LLC|Ltd|Corp|Company|Co\.))',
        ]
        
        companies = []
        for pattern in company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            companies.extend(matches)
        
        if companies:
            details['extracted_info']['company_mentions'] = companies
        
        # Check for vague company info
        vague_indicators = ['international company', 'growing company', 'established firm', 
                          'our client', 'top company', 'leading company']
        has_vague_company = any(vi in text_lower for vi in vague_indicators)
        
        if has_vague_company and not companies:
            details['detailed_advisories'].append({
                'category': 'Company Information',
                'finding': 'Vague or missing company details',
                'risk_level': 'Medium',
                'advisory': '''This posting uses vague language about the employer without providing specific company information.

**Why this matters:** Legitimate job postings typically include:
- Company name
- Company website
- Physical address or headquarters location
- Industry/sector information

**What to look for:**
1. Can you find the company on LinkedIn with employee profiles?
2. Do they have a professional website with an "About Us" page?
3. Are there reviews on Glassdoor or Indeed?
4. Can you verify their business registration?

**Be cautious if:** The posting says things like "International company" or "Leading firm" without naming the actual business.''',
                'icon': '🏢'
            })
        
        # ================================================================
        # 5. APPLICATION PROCESS ANALYSIS
        # ================================================================
        # Check for suspicious application requirements
        if re.search(r'(ssn|social\s*security|bank\s*account|credit\s*card|passport)', text_lower):
            details['extracted_info']['requests_sensitive_info'] = True
            details['detailed_advisories'].append({
                'category': 'Personal Information Request',
                'finding': 'Requests sensitive personal information',
                'risk_level': 'Critical',
                'advisory': '''⚠️ IMPORTANT: This posting requests sensitive personal information like SSN, bank details, or ID documents.

**This is a major red flag!** Legitimate employers NEVER need:
- Social Security Number before hiring
- Bank account details before you're on payroll
- Credit card information (ever)
- Passport/ID copies before an interview

**Why scammers want this:**
- Identity theft
- Financial fraud
- Creating fake accounts in your name

**What you should do:**
1. DO NOT provide any sensitive information at this stage
2. Legitimate employers only collect SSN/bank info AFTER a formal job offer
3. This information is collected through official HR systems, not email
4. Report this posting if on a job board''',
                'icon': '🔴'
            })
        
        # ================================================================
        # 6. JOB REQUIREMENTS ANALYSIS
        # ================================================================
        # No experience needed
        if re.search(r'no\s*(experience|skills?|qualifications?)\s*(needed|required|necessary)', text_lower):
            # Check if combined with high pay
            high_pay = re.search(r'\$\s*[2-9]\d{2,}.*/(hour|hr|week|wk)', text_lower)
            if high_pay:
                details['detailed_advisories'].append({
                    'category': 'Job Requirements',
                    'finding': 'No experience required + high pay promise',
                    'risk_level': 'High',
                    'advisory': '''This posting advertises high pay with "no experience required."

**Why this combination is concerning:**
- Legitimate high-paying jobs require skills and experience
- Scammers use "easy money" promises to attract victims
- Real employers invest in qualified candidates

**Questions to ask yourself:**
1. Would a real company pay this much for someone with no experience?
2. What's the catch? (Often: you'll be asked to pay for training, equipment, or "invest")
3. Does this match what similar jobs pay?

**The reality:** Most "no experience, high pay" offers lead to:
- Pyramid/MLM schemes
- Money mule operations (illegal!)
- Advance fee scams''',
                    'icon': '⚠️'
                })
        
        # ================================================================
        # 7. WORK ARRANGEMENT ANALYSIS
        # ================================================================
        if re.search(r'work\s*from\s*home|remote|wfh|home.?based', text_lower):
            details['extracted_info']['work_arrangement'] = 'remote'
            
            # Check for warning signs with remote work
            warning_signs = 0
            if re.search(r'(start|begin)\s*(today|immediately|now)', text_lower):
                warning_signs += 1
            if re.search(r'(receive|handle|process)\s*(packages?|shipments?|payments?)', text_lower):
                warning_signs += 1
            if re.search(r'use\s*your\s*(own|personal)\s*(bank|account)', text_lower):
                warning_signs += 1
            
            if warning_signs >= 1:
                details['detailed_advisories'].append({
                    'category': 'Work Arrangement',
                    'finding': 'Remote work with suspicious tasks',
                    'risk_level': 'High',
                    'advisory': '''This remote job posting contains elements commonly found in scams.

**Common remote work scams include:**

1. **Package Reshipping:** "Receive packages at home and forward them"
   - These are often stolen goods - you become complicit in theft
   
2. **Payment Processing:** "Process payments through your account"
   - This is money laundering - a serious federal crime
   
3. **Check Cashing:** "Deposit checks and wire a portion"
   - The checks are fake - you'll owe the bank the full amount

**Legitimate remote jobs:**
- Go through a proper interview process
- Have you sign employment contracts
- Pay through normal payroll (direct deposit)
- Don't ask you to use personal accounts for company business''',
                    'icon': '🏠'
                })
        
        # ================================================================
        # 8. OVERALL RISK SUMMARY
        # ================================================================
        risk_levels = [adv['risk_level'] for adv in details['detailed_advisories']]
        critical_count = risk_levels.count('Critical')
        high_count = risk_levels.count('High')
        medium_count = risk_levels.count('Medium') + risk_levels.count('Medium-High')
        
        if critical_count > 0:
            overall_risk = 'CRITICAL'
            overall_advice = 'We strongly advise against pursuing this opportunity without thorough verification.'
        elif high_count >= 2:
            overall_risk = 'HIGH'
            overall_advice = 'Multiple concerning elements detected. Please verify all details carefully before proceeding.'
        elif high_count >= 1 or medium_count >= 2:
            overall_risk = 'MEDIUM'
            overall_advice = 'Some concerning elements found. We recommend additional verification before sharing personal information.'
        else:
            overall_risk = 'LOW'
            overall_advice = 'No major red flags detected, but always verify company details before applying.'
        
        details['risk_summary'] = {
            'overall_risk': overall_risk,
            'overall_advice': overall_advice,
            'critical_findings': critical_count,
            'high_risk_findings': high_count,
            'medium_risk_findings': medium_count
        }
        
        return details
    
    def predict(self, text: str) -> Dict:
        """
        Make a prediction for a single job posting.
        
        Combines BERT model predictions with rule-based fraud signal detection
        for more robust fraud identification.
        
        Args:
            text: The job posting text to analyze
        
        Returns:
            Dictionary containing:
                - prediction: Integer label (0=legitimate, 1=fraudulent)
                - label: String label ('Legitimate' or 'Fraudulent')
                - confidence: Confidence score (0-1)
                - probabilities: Dict with probability for each class
                - fraud_signals: List of detected red flags (if any)
        
        Raises:
            ValueError: If model is not loaded
        
        Example:
            >>> result = detector.predict("Software Engineer at Google...")
            >>> print(result)
            {'prediction': 0, 'label': 'Legitimate', 'confidence': 0.95, ...}
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MODEL_CONFIG["max_length"],
            return_tensors='pt'
        )
        
        # Make prediction without computing gradients
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probabilities = outputs['probabilities'].cpu().numpy()[0]
        
        # Get rule-based fraud signals
        fraud_detection = self.detect_fraud_signals(text)
        fraud_signals = fraud_detection['signals']
        fraud_score = fraud_detection['score']
        
        # === HYBRID DETECTION STRATEGY ===
        # Rules = Definitive fraud indicators (emails, fees, gift cards)
        # BERT = Catches subtle/sophisticated scams that bypass rules
        
        bert_fraud_prob = float(probabilities[1])
        bert_legit_prob = float(probabilities[0])
        
        if fraud_score > 0:
            # RULES TRIGGERED: Definitive fraud indicators found
            # These are hard evidence (upfront fees, personal emails, etc.)
            
            if fraud_score >= 0.5:
                # Strong rule signals = definitely fraud
                # Scale confidence based on how many signals found
                final_fraud_prob = min(0.70 + (fraud_score * 0.28), 0.98)
            elif fraud_score >= 0.25:
                # Moderate signals = likely fraud
                final_fraud_prob = min(0.60 + (fraud_score * 0.3), 0.90)
            else:
                # Weak signals = suspicious, combine with BERT
                # If BERT also suspects fraud, increase confidence
                final_fraud_prob = max(0.55, bert_fraud_prob + (fraud_score * 0.2))
            
            final_legit_prob = 1 - final_fraud_prob
        else:
            # NO RULES TRIGGERED: Rely on BERT for sophisticated scams
            # BERT catches subtle patterns like:
            # - Unusual language combinations
            # - Vague job descriptions that sound professional
            # - Hidden scam patterns learned from training data
            final_fraud_prob = bert_fraud_prob
            final_legit_prob = bert_legit_prob
        
        # Get the predicted class based on adjusted probabilities
        if final_fraud_prob > final_legit_prob:
            predicted_class = 1  # Fraudulent
            confidence = final_fraud_prob
        else:
            predicted_class = 0  # Legitimate
            confidence = final_legit_prob
        
        # Build result dictionary (convert to percentages for display)
        result = {
            'prediction': predicted_class,
            'label': LABEL_MAP[predicted_class],
            'confidence': round(float(confidence) * 100, 2),
            'probabilities': {
                'legitimate': round(float(final_legit_prob) * 100, 2),
                'fraudulent': round(float(final_fraud_prob) * 100, 2)
            },
            # Store BERT's raw prediction for transparency
            'bert_raw': {
                'legitimate': round(bert_legit_prob * 100, 2),
                'fraudulent': round(bert_fraud_prob * 100, 2)
            }
        }
        
        # Add fraud signals if detected
        if fraud_signals:
            result['fraud_signals'] = fraud_signals
        
        return result
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for multiple texts.
        
        Args:
            texts: List of job posting texts
        
        Returns:
            2D numpy array of shape (n_samples, 2) with probabilities
        
        Raises:
            ValueError: If model is not loaded
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Tokenize all texts
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MODEL_CONFIG["max_length"],
            return_tensors='pt'
        )
        
        # Create a DataLoader for batch processing
        dataset = torch.utils.data.TensorDataset(
            encoding['input_ids'],
            encoding['attention_mask']
        )
        loader = DataLoader(dataset, batch_size=MODEL_CONFIG["batch_size"])
        
        # Collect probabilities
        all_probabilities = []
        
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device)
                )
                all_probabilities.extend(outputs['probabilities'].cpu().numpy())
        
        return np.array(all_probabilities)
    
    def save_model(self, path: Path = None):
        """
        Save the trained model to disk.
        
        Args:
            path: File path for saving (default: MODEL_DIR/model.pt)
        """
        # Use default path if not specified
        if path is None:
            path = MODEL_DIR / "model.pt"
        else:
            path = Path(path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state, config, and training history
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': MODEL_CONFIG,
            'training_history': self.training_history
        }
        torch.save(checkpoint, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path = None):
        """
        Load a trained model from disk.
        
        Args:
            path: File path to load from (default: MODEL_DIR/best_model.pt)
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        # Use default path if not specified
        if path is None:
            path = MODEL_DIR / "best_model.pt"
        else:
            path = Path(path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Model not found at: {path}")
        
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Create a new model and load the saved weights
        self.model = BertClassifier()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Restore training history if available
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        logger.info(f"Model loaded from {path}")

    def predict_from_image(
        self,
        image_source,
        tesseract_path: Optional[str] = None,
        preprocess: bool = True
    ) -> Dict:
        """
        Extract text from an image and predict if it's a fraudulent job posting.
        
        This method combines OCR text extraction with the BERT-based prediction
        for end-to-end image-based fraud detection.
        
        Args:
            image_source: File path (str/Path) or PIL Image object
            tesseract_path: Optional path to Tesseract executable (auto-detected if None)
            preprocess: Whether to apply image preprocessing for better OCR (default: True)
        
        Returns:
            Dictionary containing:
                - prediction: Integer label (0=legitimate, 1=fraudulent)
                - label: String label ('Legitimate' or 'Fraudulent')
                - confidence: Confidence score (0-100)
                - probabilities: Dict with probability for each class
                - fraud_signals: List of detected red flags (if any)
                - extracted_text: The OCR-extracted text
                - text_length: Character count of extracted text
                - source_type: 'image'
        
        Raises:
            ValueError: If model not loaded or no text extracted
            FileNotFoundError: If image file doesn't exist
            ImportError: If OCR dependencies not installed
        
        Example:
            >>> detector = JobFraudDetector()
            >>> detector.load_model()
            >>> result = detector.predict_from_image("job_screenshot.png")
            >>> print(f"{result['label']} ({result['confidence']}% confidence)")
        """
        try:
            from .image_ocr import ImageOCR
        except ImportError as e:
            raise ImportError(
                "Image OCR dependencies not installed. Install with:\n"
                "pip install pytesseract opencv-python Pillow\n"
                f"Original error: {e}"
            )
        
        ocr = ImageOCR(tesseract_path=tesseract_path, preprocess=preprocess)
        return ocr.predict_from_image(image_source, self)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_demo_model() -> JobFraudDetector:
    """
    Create and save a demo model for testing purposes.
    
    This function creates an untrained BERT model and saves it.
    Useful for setting up the application without training data.
    
    Returns:
        JobFraudDetector instance with initialized (but untrained) model
    """
    detector = JobFraudDetector()
    detector.initialize_model()
    detector.save_model(MODEL_DIR / "best_model.pt")
    logger.info("Demo model created and saved")
    return detector

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = create_demo_model()
    print(detector.predict("Software Engineer at Google. Competitive salary."))
