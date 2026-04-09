"""
Source modules initialization
Note: Imports are lazy to avoid transformers compatibility issues
"""
# Import BERT-based model
from .model import JobFraudDetector, BertClassifier, JobPostingDataset
from .explainer import LIMEExplainer

# These require transformers and may fail on Python 3.13
# from .preprocessing import DataPreprocessor, TextPreprocessorForInference

def get_preprocessor():
    """Lazy import of DataPreprocessor to handle transformers issues."""
    from .preprocessing import DataPreprocessor
    return DataPreprocessor

def get_bert_classifier():
    """Lazy import of BertClassifier to handle transformers issues."""
    from .model import BertClassifier
    return BertClassifier

def get_image_ocr():
    """
    Lazy import of ImageOCR to handle optional dependencies.
    
    Returns:
        ImageOCR class for image-based fraud detection
    
    Raises:
        ImportError: If pytesseract or opencv-python is not installed
    """
    from .image_ocr import ImageOCR
    return ImageOCR

def get_predict_from_image():
    """
    Lazy import of predict_from_image function.
    
    Returns:
        predict_from_image function for quick image-based predictions
    """
    from .image_ocr import predict_from_image
    return predict_from_image
