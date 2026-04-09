"""
=============================================================================
Image OCR Module for Fake Job Detection
=============================================================================

This module provides OCR (Optical Character Recognition) capabilities to
extract text from job posting images (screenshots, photos, etc.) and 
analyze them using the existing JobFraudDetector.

Main Components:
    1. ImagePreprocessor - Handles image preprocessing for better OCR accuracy
    2. ImageOCR - Main class for text extraction and fraud detection from images

Dependencies:
    - pytesseract: Python wrapper for Tesseract OCR
    - opencv-python: Image preprocessing
    - Pillow: Image loading and manipulation

Installation Notes:
    - Tesseract OCR must be installed separately on the system
    - Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
    - Linux: sudo apt-get install tesseract-ocr
    - macOS: brew install tesseract

Author: ORFD Project Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

import re
import logging
from pathlib import Path
from typing import Union, Dict, Optional, Tuple
import sys

# Image processing
try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
except ImportError as e:
    raise ImportError(
        f"Missing required dependencies for image OCR: {e}\n"
        "Install with: pip install pytesseract opencv-python Pillow\n"
        "Also ensure Tesseract OCR is installed on your system."
    )

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# IMAGE PREPROCESSOR
# =============================================================================

class ImagePreprocessor:
    """
    Handles image preprocessing to improve OCR accuracy.
    
    Preprocessing steps:
        1. Convert to grayscale
        2. Apply thresholding (adaptive or Otsu)
        3. Remove noise (morphological operations)
        4. Optional: Deskewing, scaling
    
    Example:
        >>> preprocessor = ImagePreprocessor()
        >>> processed_img = preprocessor.preprocess(image)
    """
    
    def __init__(
        self,
        apply_grayscale: bool = True,
        apply_threshold: bool = True,
        apply_denoise: bool = True,
        threshold_method: str = "otsu"
    ):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            apply_grayscale: Convert image to grayscale
            apply_threshold: Apply thresholding for better text contrast
            apply_denoise: Apply noise removal
            threshold_method: Thresholding method ("otsu" or "adaptive")
        """
        self.apply_grayscale = apply_grayscale
        self.apply_threshold = apply_threshold
        self.apply_denoise = apply_denoise
        self.threshold_method = threshold_method
        
        logger.info(
            f"ImagePreprocessor initialized: grayscale={apply_grayscale}, "
            f"threshold={apply_threshold}, denoise={apply_denoise}"
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image as numpy array (BGR or grayscale)
        
        Returns:
            Preprocessed image as numpy array
        """
        processed = image.copy()
        
        # Step 1: Convert to grayscale
        if self.apply_grayscale:
            processed = self._to_grayscale(processed)
        
        # Step 2: Remove noise
        if self.apply_denoise:
            processed = self._remove_noise(processed)
        
        # Step 3: Apply thresholding
        if self.apply_threshold:
            processed = self._apply_threshold(processed)
        
        return processed
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image (BGR or already grayscale)
        
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # Handle RGBA images
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image using morphological operations.
        
        Args:
            image: Input grayscale image
        
        Returns:
            Denoised image
        """
        # Apply Gaussian blur to reduce high-frequency noise
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply median blur for salt-and-pepper noise
        denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    def _apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply thresholding to binarize the image.
        
        Args:
            image: Input grayscale image
        
        Returns:
            Thresholded (binary) image
        """
        if self.threshold_method == "adaptive":
            # Adaptive threshold works well for varying lighting
            return cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # Block size
                2    # Constant subtracted from mean
            )
        else:
            # Otsu's method automatically determines optimal threshold
            _, thresholded = cv2.threshold(
                image,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresholded
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew (straighten) a rotated image.
        
        Useful for scanned documents or photos taken at an angle.
        
        Args:
            image: Input grayscale image
        
        Returns:
            Deskewed image
        """
        # Find all non-zero points (text pixels)
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) < 10:
            return image
        
        # Find the angle of rotation
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        # Only deskew if angle is significant
        if abs(angle) < 0.5:
            return image
        
        # Rotate the image to correct the skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def resize_for_ocr(
        self, 
        image: np.ndarray, 
        target_dpi: int = 300
    ) -> np.ndarray:
        """
        Resize image to optimal size for OCR.
        
        Tesseract works best with images at 300 DPI or higher.
        
        Args:
            image: Input image
            target_dpi: Target DPI for OCR (default: 300)
        
        Returns:
            Resized image
        """
        # Estimate current DPI based on typical screen capture
        # Most screenshots are around 72-96 DPI
        current_dpi = 96
        scale_factor = target_dpi / current_dpi
        
        # Only upscale if needed
        if scale_factor > 1:
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            return cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_CUBIC
            )
        
        return image


# =============================================================================
# IMAGE OCR CLASS
# =============================================================================

class ImageOCR:
    """
    Main class for OCR-based text extraction from job posting images.
    
    This class provides:
        - Image loading from file path or PIL Image
        - Full preprocessing pipeline for OCR optimization
        - Text extraction using Tesseract OCR
        - Integration with JobFraudDetector for fraud analysis
    
    Example:
        >>> from src.model import JobFraudDetector
        >>> ocr = ImageOCR()
        >>> detector = JobFraudDetector()
        >>> detector.load_model()
        >>> result = ocr.predict_from_image("job_screenshot.png", detector)
        >>> print(result['label'], result['confidence'])
    """
    
    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        preprocess: bool = True,
        language: str = "eng"
    ):
        """
        Initialize the ImageOCR.
        
        Args:
            tesseract_path: Path to Tesseract executable (auto-detected if None)
            preprocess: Whether to apply image preprocessing
            language: OCR language code (default: "eng" for English)
        """
        # Auto-detect Tesseract path on Windows if not provided
        if tesseract_path is None:
            import platform
            if platform.system() == "Windows":
                windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if Path(windows_path).exists():
                    tesseract_path = windows_path
        
        # Configure Tesseract path if provided or detected
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Verify Tesseract is accessible
        self._verify_tesseract()
        
        self.preprocess = preprocess
        self.language = language
        self.preprocessor = ImagePreprocessor() if preprocess else None
        
        logger.info(f"ImageOCR initialized: preprocess={preprocess}, language={language}")
    
    def _verify_tesseract(self):
        """
        Verify that Tesseract OCR is installed and accessible.
        
        Raises:
            EnvironmentError: If Tesseract is not found
        """
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise EnvironmentError(
                "Tesseract OCR is not installed or not in PATH.\n\n"
                "Installation instructions:\n"
                "- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  After installation, add to PATH or set tesseract_path parameter.\n"
                "- Linux: sudo apt-get install tesseract-ocr\n"
                "- macOS: brew install tesseract\n\n"
                "For Windows, common installation path:\n"
                "  C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            )
    
    def load_image(
        self, 
        image_source: Union[str, Path, Image.Image]
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Load an image from various sources.
        
        Args:
            image_source: File path (str/Path) or PIL Image object
        
        Returns:
            Tuple of (OpenCV image as numpy array, PIL Image)
        
        Raises:
            ValueError: If image source is invalid
            FileNotFoundError: If file path doesn't exist
        """
        if isinstance(image_source, (str, Path)):
            # Load from file path
            path = Path(image_source)
            
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            
            if not path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif']:
                raise ValueError(f"Unsupported image format: {path.suffix}")
            
            # Load with PIL first (handles more formats)
            pil_image = Image.open(path)
            
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if pil_image.mode not in ['RGB', 'L']:
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format
            cv_image = np.array(pil_image)
            if len(cv_image.shape) == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Loaded image from file: {path} ({pil_image.size[0]}x{pil_image.size[1]})")
            
        elif isinstance(image_source, Image.Image):
            # Already a PIL Image
            pil_image = image_source
            
            if pil_image.mode not in ['RGB', 'L']:
                pil_image = pil_image.convert('RGB')
            
            cv_image = np.array(pil_image)
            if len(cv_image.shape) == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Loaded PIL Image: {pil_image.size[0]}x{pil_image.size[1]}")
            
        else:
            raise ValueError(
                f"Invalid image source type: {type(image_source)}. "
                "Expected file path (str/Path) or PIL Image."
            )
        
        return cv_image, pil_image
    
    def extract_text(
        self, 
        image_source: Union[str, Path, Image.Image],
        config: str = "--oem 3 --psm 6"
    ) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_source: File path or PIL Image
            config: Tesseract configuration string
                    --oem 3: Use LSTM OCR Engine Mode (default, best accuracy)
                    --psm 6: Assume uniform block of text
                    Other PSM modes:
                        3: Fully automatic page segmentation
                        4: Single column of text
                        6: Uniform block of text (default)
                        11: Sparse text
        
        Returns:
            Extracted text as string
        
        Raises:
            ValueError: If no text could be extracted
        """
        # Load image
        cv_image, pil_image = self.load_image(image_source)
        
        # Apply preprocessing if enabled
        if self.preprocess and self.preprocessor:
            processed_image = self.preprocessor.preprocess(cv_image)
            # Convert back to PIL for pytesseract
            ocr_image = Image.fromarray(processed_image)
        else:
            ocr_image = pil_image
        
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(
            ocr_image,
            lang=self.language,
            config=config
        )
        
        # Clean up extracted text
        extracted_text = self._clean_extracted_text(extracted_text)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            raise ValueError(
                "No text could be extracted from the image. "
                "Please ensure the image contains readable text."
            )
        
        logger.info(f"Extracted {len(extracted_text)} characters from image")
        
        return extracted_text
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize OCR-extracted text.
        
        Performs:
            - Lowercase conversion
            - Remove excessive whitespace
            - Fix common OCR errors
            - Remove non-printable characters
        
        Args:
            text: Raw OCR output
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-printable characters (except newlines and spaces)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        
        # Fix common OCR errors
        ocr_fixes = {
            r'\bl\b': 'I',      # Lowercase L often misread as I
            r'\bO\b': '0',      # Capital O often misread as 0
            r'rn\b': 'm',       # 'rn' often misread as 'm'
            r'\|': 'I',         # Pipe often misread as I
        }
        # Note: We're keeping text lowercase, so these fixes are minimal
        
        # Normalize whitespace (multiple spaces/tabs to single space)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize multiple newlines to double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Final strip
        text = text.strip()
        
        return text
    
    def predict_from_image(
        self,
        image_source: Union[str, Path, Image.Image],
        detector
    ) -> Dict:
        """
        Extract text from an image and predict if it's a fraudulent job posting.
        
        This method combines OCR text extraction with the existing
        JobFraudDetector.predict() method for end-to-end image-based
        fraud detection.
        
        Args:
            image_source: File path (str/Path) or PIL Image object
            detector: An initialized JobFraudDetector instance with loaded model
        
        Returns:
            Dictionary containing:
                - All fields from detector.predict():
                    - prediction: Integer label (0=legitimate, 1=fraudulent)
                    - label: String label ('Legitimate' or 'Fraudulent')
                    - confidence: Confidence score (0-100)
                    - probabilities: Dict with probability for each class
                    - fraud_signals: List of detected red flags (if any)
                - Additional fields:
                    - extracted_text: The OCR-extracted text
                    - text_length: Character count of extracted text
                    - source_type: 'image'
        
        Raises:
            ValueError: If detector has no model loaded or no text extracted
            FileNotFoundError: If image file doesn't exist
        
        Example:
            >>> from src.model import JobFraudDetector
            >>> from src.image_ocr import ImageOCR
            >>> 
            >>> # Initialize detector and load model
            >>> detector = JobFraudDetector()
            >>> detector.load_model()
            >>> 
            >>> # Initialize OCR
            >>> ocr = ImageOCR()
            >>> 
            >>> # Predict from image
            >>> result = ocr.predict_from_image("job_posting.png", detector)
            >>> 
            >>> print(f"Label: {result['label']}")
            >>> print(f"Confidence: {result['confidence']}%")
            >>> print(f"Extracted text preview: {result['extracted_text'][:100]}...")
        """
        # Validate detector
        if not hasattr(detector, 'predict'):
            raise ValueError(
                "Invalid detector. Expected JobFraudDetector instance with predict() method."
            )
        
        if not hasattr(detector, 'model') or detector.model is None:
            raise ValueError(
                "Detector model not loaded. Call detector.load_model() first."
            )
        
        # Extract text from image
        try:
            extracted_text = self.extract_text(image_source)
        except ValueError as e:
            # No text extracted
            return {
                'prediction': -1,
                'label': 'Unable to Analyze',
                'confidence': 0.0,
                'probabilities': {'legitimate': 0.0, 'fraudulent': 0.0},
                'error': str(e),
                'extracted_text': '',
                'text_length': 0,
                'source_type': 'image'
            }
        
        # Get prediction from the existing detector
        result = detector.predict(extracted_text)
        
        # Add OCR-specific metadata
        result['extracted_text'] = extracted_text
        result['text_length'] = len(extracted_text)
        result['source_type'] = 'image'
        
        # Optionally add posting details extraction
        if hasattr(detector, 'extract_posting_details'):
            result['posting_details'] = detector.extract_posting_details(extracted_text)
        
        logger.info(
            f"Image prediction: {result['label']} "
            f"(confidence: {result['confidence']}%)"
        )
        
        return result
    
    def predict_from_image_detailed(
        self,
        image_source: Union[str, Path, Image.Image],
        detector,
        include_lime: bool = True,
        lime_num_features: int = 10
    ) -> Dict:
        """
        Extract text from an image and provide detailed fraud analysis.
        
        This method extends predict_from_image() with comprehensive analysis:
            - LIME explanations showing which words influenced the prediction
            - Full posting details extraction with advisories
            - OCR confidence scores 
            - Risk categorization
        
        Args:
            image_source: File path (str/Path) or PIL Image object
            detector: An initialized JobFraudDetector instance with loaded model
            include_lime: Whether to include LIME explanations (default: True)
            lime_num_features: Number of features for LIME (default: 10)
        
        Returns:
            Dictionary containing:
                - All fields from predict_from_image()
                - lime_explanation: Dict with word-level explanations
                - posting_details: Comprehensive analysis with advisories
                - ocr_confidence: OCR quality assessment
                - analysis_summary: Human-readable summary
        
        Example:
            >>> result = ocr.predict_from_image_detailed("job.png", detector)
            >>> print(result['analysis_summary'])
            >>> for adv in result['posting_details']['detailed_advisories']:
            ...     print(f"[{adv['risk_level']}] {adv['finding']}")
        """
        # Validate detector
        if not hasattr(detector, 'predict'):
            raise ValueError(
                "Invalid detector. Expected JobFraudDetector instance with predict() method."
            )
        
        if not hasattr(detector, 'model') or detector.model is None:
            raise ValueError(
                "Detector model not loaded. Call detector.load_model() first."
            )
        
        # Extract text with confidence scores
        try:
            ocr_details = self.get_text_with_confidence(image_source)
            extracted_text = ocr_details['text']
            ocr_confidence = ocr_details['average_confidence']
        except Exception as e:
            # Fallback to simple extraction
            try:
                extracted_text = self.extract_text(image_source)
                ocr_confidence = None
            except ValueError as ve:
                return {
                    'prediction': -1,
                    'label': 'Unable to Analyze',
                    'confidence': 0.0,
                    'probabilities': {'legitimate': 0.0, 'fraudulent': 0.0},
                    'error': str(ve),
                    'extracted_text': '',
                    'text_length': 0,
                    'source_type': 'image'
                }
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            return {
                'prediction': -1,
                'label': 'Unable to Analyze',
                'confidence': 0.0,
                'probabilities': {'legitimate': 0.0, 'fraudulent': 0.0},
                'error': 'Insufficient text extracted from image',
                'extracted_text': extracted_text,
                'text_length': len(extracted_text) if extracted_text else 0,
                'source_type': 'image'
            }
        
        # Get base prediction
        result = detector.predict(extracted_text)
        
        # Add OCR metadata
        result['extracted_text'] = extracted_text
        result['text_length'] = len(extracted_text)
        result['source_type'] = 'image'
        
        # Add OCR quality assessment
        result['ocr_confidence'] = {
            'average_score': ocr_confidence,
            'quality': self._assess_ocr_quality(ocr_confidence)
        }
        
        # Get detailed posting analysis
        if hasattr(detector, 'extract_posting_details'):
            result['posting_details'] = detector.extract_posting_details(extracted_text)
        
        # Add LIME explanation
        if include_lime:
            try:
                from .explainer import LIMEExplainer
                
                explainer = LIMEExplainer(num_features=lime_num_features)
                lime_result = explainer.explain(
                    extracted_text,
                    detector.predict_proba,
                    num_features=lime_num_features
                )
                
                result['lime_explanation'] = {
                    'interpretation': lime_result.get('interpretation', ''),
                    'key_indicators': [
                        {
                            'word': f['word'],
                            'impact': 'supports prediction' if f['weight'] > 0 else 'opposes prediction',
                            'weight': round(f['weight'], 4)
                        }
                        for f in lime_result.get('positive_features', [])[:5]
                    ],
                    'opposing_indicators': [
                        {
                            'word': f['word'],
                            'impact': 'opposes prediction',
                            'weight': round(abs(f['weight']), 4)
                        }
                        for f in lime_result.get('negative_features', [])[:3]
                    ]
                }
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
                result['lime_explanation'] = {
                    'error': str(e),
                    'interpretation': 'LIME explanation unavailable'
                }
        
        # Generate comprehensive analysis summary
        result['analysis_summary'] = self._generate_analysis_summary(result)
        
        logger.info(
            f"Detailed image prediction: {result['label']} "
            f"(confidence: {result['confidence']}%)"
        )
        
        return result
    
    def _assess_ocr_quality(self, confidence: Optional[float]) -> str:
        """Assess OCR quality based on confidence score."""
        if confidence is None:
            return 'unknown'
        elif confidence >= 85:
            return 'excellent'
        elif confidence >= 70:
            return 'good'
        elif confidence >= 50:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_analysis_summary(self, result: Dict) -> str:
        """Generate a human-readable analysis summary."""
        summary_parts = []
        
        label = result.get('label', 'Unknown')
        confidence = result.get('confidence', 0)
        
        # Overall assessment
        if label == 'Fraudulent':
            summary_parts.append(
                f"⚠️ ALERT: This job posting appears to be FRAUDULENT "
                f"with {confidence}% confidence."
            )
        else:
            summary_parts.append(
                f"✅ This job posting appears to be LEGITIMATE "
                f"with {confidence}% confidence."
            )
        
        # Fraud signals
        fraud_signals = result.get('fraud_signals', [])
        if fraud_signals:
            summary_parts.append(f"\n🚨 Red Flags Detected ({len(fraud_signals)}):")
            for signal in fraud_signals:
                summary_parts.append(f"   • {signal}")
        
        # Risk summary
        if 'posting_details' in result:
            details = result['posting_details']
            if 'risk_summary' in details:
                risk = details['risk_summary']
                summary_parts.append(f"\n⚖️ Risk Level: {risk.get('overall_risk', 'Unknown')}")
                
                if risk.get('critical_findings', 0) > 0:
                    summary_parts.append(f"   • {risk['critical_findings']} critical issue(s)")
                if risk.get('high_risk_findings', 0) > 0:
                    summary_parts.append(f"   • {risk['high_risk_findings']} high-risk issue(s)")
            
            # Key advisories
            advisories = details.get('detailed_advisories', [])
            if advisories:
                summary_parts.append(f"\n📋 Key Findings ({len(advisories)}):")
                for adv in advisories[:3]:  # Show top 3
                    icon = adv.get('icon', '•')
                    summary_parts.append(
                        f"   {icon} [{adv['risk_level']}] {adv['finding']}"
                    )
                if len(advisories) > 3:
                    summary_parts.append(f"   ... and {len(advisories) - 3} more")
        
        # LIME insights
        if 'lime_explanation' in result and 'key_indicators' in result['lime_explanation']:
            indicators = result['lime_explanation']['key_indicators']
            if indicators:
                summary_parts.append("\n🔍 Key Words Influencing Analysis:")
                for ind in indicators[:3]:
                    summary_parts.append(f"   • \"{ind['word']}\"")
        
        # OCR quality note
        ocr_conf = result.get('ocr_confidence', {})
        if ocr_conf.get('quality') in ['fair', 'poor']:
            summary_parts.append(
                f"\n⚠️ Note: OCR quality is {ocr_conf['quality']}. "
                "Some text may not have been extracted accurately."
            )
        
        return '\n'.join(summary_parts)
    
    def get_text_with_confidence(
        self,
        image_source: Union[str, Path, Image.Image]
    ) -> Dict:
        """
        Extract text with per-word confidence scores.
        
        Useful for debugging OCR quality and identifying
        low-confidence extractions.
        
        Args:
            image_source: File path or PIL Image
        
        Returns:
            Dictionary with:
                - text: Full extracted text
                - word_details: List of dicts with word, confidence, position
                - average_confidence: Overall OCR confidence (0-100)
        """
        # Load and preprocess image
        cv_image, pil_image = self.load_image(image_source)
        
        if self.preprocess and self.preprocessor:
            processed_image = self.preprocessor.preprocess(cv_image)
            ocr_image = Image.fromarray(processed_image)
        else:
            ocr_image = pil_image
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            ocr_image,
            lang=self.language,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract word details
        word_details = []
        confidences = []
        
        for i, word in enumerate(data['text']):
            if word.strip():
                conf = data['conf'][i]
                if conf > 0:  # -1 means no confidence available
                    word_details.append({
                        'word': word,
                        'confidence': conf,
                        'position': {
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
                    confidences.append(conf)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'text': self._clean_extracted_text(' '.join([w['word'] for w in word_details])),
            'word_details': word_details,
            'average_confidence': round(avg_confidence, 2),
            'total_words': len(word_details)
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def predict_from_image(
    image_path: Union[str, Path],
    detector,
    tesseract_path: Optional[str] = None,
    preprocess: bool = True
) -> Dict:
    """
    Convenience function to predict fraud from a job posting image.
    
    This is a simple wrapper around ImageOCR.predict_from_image() for
    quick one-off predictions without explicitly creating an ImageOCR instance.
    
    Args:
        image_path: Path to the job posting image
        detector: Initialized JobFraudDetector instance with loaded model
        tesseract_path: Optional path to Tesseract executable
        preprocess: Whether to apply image preprocessing (default: True)
    
    Returns:
        Dictionary with prediction results (see ImageOCR.predict_from_image)
    
    Example:
        >>> from src.model import JobFraudDetector
        >>> from src.image_ocr import predict_from_image
        >>> 
        >>> detector = JobFraudDetector()
        >>> detector.load_model()
        >>> 
        >>> result = predict_from_image("job_screenshot.png", detector)
        >>> print(f"{result['label']} ({result['confidence']}% confidence)")
    """
    ocr = ImageOCR(tesseract_path=tesseract_path, preprocess=preprocess)
    return ocr.predict_from_image(image_path, detector)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example demonstrating image-based fraud detection.
    
    Run this script directly to see a demonstration:
        python -m src.image_ocr
    """
    import os
    
    print("=" * 60)
    print("Image OCR for Fake Job Detection - Demo")
    print("=" * 60)
    
    # Check if Tesseract is installed
    try:
        version = pytesseract.get_tesseract_version()
        print(f"\n✓ Tesseract OCR version: {version}")
    except pytesseract.TesseractNotFoundError:
        print("\n✗ Tesseract OCR not found!")
        print("\nInstallation instructions:")
        print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("- Linux: sudo apt-get install tesseract-ocr")
        print("- macOS: brew install tesseract")
        exit(1)
    
    # Import detector
    from model import JobFraudDetector
    from pathlib import Path
    
    # Initialize
    print("\n1. Initializing JobFraudDetector...")
    detector = JobFraudDetector()
    
    # Load model
    print("2. Loading trained model...")
    try:
        detector.load_model()
        print("   ✓ Model loaded successfully")
    except FileNotFoundError:
        print("   ✗ Model not found. Please train the model first:")
        print("     python train_bert.py")
        exit(1)
    
    # Initialize OCR
    print("3. Initializing ImageOCR...")
    ocr = ImageOCR()
    print("   ✓ OCR ready")
    
    # Example usage with a test image (if exists)
    test_images = list(Path("data").glob("*.png")) + list(Path("data").glob("*.jpg"))
    
    if test_images:
        test_image = test_images[0]
        print(f"\n4. Testing with image: {test_image}")
        print("-" * 40)
        
        try:
            result = ocr.predict_from_image(test_image, detector)
            
            print(f"\n📊 Analysis Result:")
            print(f"   Label: {result['label']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Probabilities: {result['probabilities']}")
            
            if 'fraud_signals' in result:
                print(f"\n⚠️ Fraud Signals Detected:")
                for signal in result['fraud_signals']:
                    print(f"   - {signal}")
            
            print(f"\n📝 Extracted Text ({result['text_length']} chars):")
            preview = result['extracted_text'][:300]
            print(f"   {preview}...")
            
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("\n4. No test images found in data/ directory.")
        print("   To test, place a job posting screenshot in the data/ directory.")
    
    print("\n" + "=" * 60)
    print("Usage Example Code:")
    print("=" * 60)
    print("""
from src.model import JobFraudDetector
from src.image_ocr import ImageOCR, predict_from_image

# Method 1: Using ImageOCR class
detector = JobFraudDetector()
detector.load_model()

ocr = ImageOCR()
result = ocr.predict_from_image("job_screenshot.png", detector)

# Method 2: Using convenience function
result = predict_from_image("job_screenshot.png", detector)

# Access results
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}%")
print(f"Extracted Text: {result['extracted_text'][:200]}...")
""")
