"""
=============================================================================
Flask Backend API for Fake Job Detection
=============================================================================

This module provides the REST API backend for the fake job detection system.
It handles incoming requests from the frontend and returns predictions.

Available Endpoints:
    - GET  /api/health       : Check if the server is running
    - POST /api/predict      : Get a quick prediction for job posting text
    - POST /api/explain      : Get prediction with detailed explanation
    - POST /api/batch-predict: Predict multiple job postings at once
    - GET  /api/model-info   : Get information about the model
    - GET  /api/sample-jobs  : Get example job postings for testing

Author: ORFD Project Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import re
import time
import logging
import traceback
from pathlib import Path
from functools import wraps

# Add parent directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Flask - Web framework
from flask import Flask, request, jsonify
from flask_cors import CORS

# Local imports
from src.model import JobFraudDetector
from config.settings import FLASK_CONFIG, MODEL_DIR, LOGGING_CONFIG, LABEL_MAP


# =============================================================================
# LOGGING SETUP
# =============================================================================

# Configure logging format and level
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


# =============================================================================
# FLASK APP SETUP
# =============================================================================

# Create Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
# This allows the frontend to make requests to this backend
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global variable to hold the fraud detector model
detector = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize input text before processing.
    
    This function removes:
        - HTML tags (e.g., <p>, <div>)
        - URLs (http://, www.)
        - Extra whitespace
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text string
    
    Example:
        >>> clean_text("<p>Visit http://example.com for more</p>")
        "Visit for more"
    """
    # Return empty string if no text
    if not text:
        return ""
    
    # Convert to string (in case of other types)
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def initialize_services():
    """
    Initialize the machine learning services.
    
    This function:
        1. Creates a JobFraudDetector instance
        2. Loads the trained model if available
        3. Falls back to demo mode if no model exists
    
    This should be called once when the server starts.
    """
    global detector
    
    logger.info("Initializing ML services...")
    
    # Create the fraud detector
    detector = JobFraudDetector()
    
    # Path to the trained model
    model_path = MODEL_DIR / "best_model.pt"
    
    # Try to load the trained model
    if model_path.exists():
        try:
            detector.load_model(model_path)
            logger.info("Successfully loaded BERT model")
        except Exception as error:
            logger.warning(f"Failed to load model: {error}. Using demo mode.")
            detector.initialize_model()
    else:
        logger.info("No trained model found. Using demo mode.")
        detector.initialize_model()


def handle_errors(function):
    """
    Decorator to handle errors in API endpoints.
    
    This wraps an endpoint function and catches any exceptions,
    returning a proper JSON error response instead of crashing.
    
    Args:
        function: The endpoint function to wrap
    
    Returns:
        Wrapped function with error handling
    """
    @wraps(function)
    def decorated_function(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            # Log the full error with traceback
            logger.error(f"Error in {function.__name__}: {error}\n{traceback.format_exc()}")
            
            # Return error response
            return jsonify({
                'success': False,
                'error': str(error)
            }), 500
    
    return decorated_function


# =============================================================================
# KEYWORD DICTIONARIES FOR EXPLANATIONS
# =============================================================================

# Keywords that indicate potential fraud
# Format: 'keyword': ('description', severity_score)
# Severity: 1-5, where 5 is most suspicious
FRAUD_KEYWORDS = {
    # Financial red flags (highest severity)
    'send ssn': ('Requests SSN', 5),
    'bank details': ('Requests banking info', 5),
    'credit card': ('Requests credit card', 5),
    'wire money': ('Requests money transfer', 5),
    'processing fee': ('Upfront fees required', 4.5),
    'gift card': ('Gift card scam indicator', 4.5),
    
    # Too-good-to-be-true promises
    'earn money fast': ('Quick money promise', 4),
    '$5000': ('Unrealistic salary claims', 4),
    'easy money': ('Easy money claims', 3.5),
    'guaranteed': ('Guarantee claims', 2.5),
    
    # Lack of professional standards
    'no interview': ('No interview required', 4),
    'no experience': ('No experience required', 3),
    
    # Urgency and pressure tactics
    'urgent': ('Urgency tactics', 2.5),
    '!!!': ('Excessive punctuation', 2),
    
    # Informal contact methods
    'telegram': ('Informal contact method', 2),
    'gmail.com': ('Free email service', 2.5),
    'work from home': ('WFH (can be misused)', 1.5),
}

# Keywords that indicate legitimate postings
# These have negative weights (toward legitimate)
LEGITIMATE_KEYWORDS = {
    # Professional requirements
    'experience required': ('Experience requirements', 4),
    'bachelor': ('Education requirements', 4),
    'degree': ('Degree mentioned', 2.5),
    'qualifications': ('Qualification requirements', 3),
    
    # Standard benefits
    'health insurance': ('Standard benefits', 4.5),
    '401k': ('Retirement benefits', 4.5),
    'pto': ('Paid time off', 4),
    
    # Professional language
    'competitive salary': ('Professional language', 3),
    'interview': ('Interview process', 3),
    'resume': ('Resume required', 2),
}


def generate_explanation(text: str, prediction: dict) -> dict:
    """
    Generate a human-readable explanation of the prediction.
    
    This function analyzes the text for known keywords and generates
    an explanation of why the model made its prediction.
    
    Args:
        text: The job posting text
        prediction: The model's prediction dictionary
    
    Returns:
        Dictionary containing:
            - feature_contributions: List of found features with weights
            - positive_features: Features indicating fraud
            - negative_features: Features indicating legitimacy
            - interpretation: Human-readable explanation text
            - chart_data: Data for visualization charts
    """
    # Convert text to lowercase for keyword matching
    text_lower = text.lower()
    
    # Lists to store found keywords
    fraud_features = []
    legit_features = []
    
    # Search for fraud keywords
    for keyword, (description, severity) in FRAUD_KEYWORDS.items():
        if keyword in text_lower:
            # Calculate weight (positive = toward fraud)
            weight = round((0.1 + severity / 5 * 0.4), 3)
            fraud_features.append({
                'word': keyword,
                'weight': weight,
                'description': description
            })
    
    # Search for legitimate keywords
    for keyword, (description, severity) in LEGITIMATE_KEYWORDS.items():
        if keyword in text_lower:
            # Calculate weight (negative = toward legitimate)
            weight = round(-(0.1 + severity / 5 * 0.4), 3)
            legit_features.append({
                'word': keyword,
                'weight': weight,
                'description': description
            })
    
    # Sort by weight (most impactful first)
    fraud_features.sort(key=lambda x: -x['weight'])
    legit_features.sort(key=lambda x: x['weight'])
    
    # Get prediction details
    is_fraud = prediction['prediction'] == 1
    confidence = prediction['confidence']
    
    # Build interpretation text
    if is_fraud:
        interpretation = (
            f"This job posting is classified as **Fraudulent** "
            f"with {confidence:.1f}% confidence.\n\n"
        )
        
        if fraud_features:
            interpretation += "**Suspicious indicators:**\n"
            for feature in fraud_features[:5]:
                interpretation += f"- '{feature['word']}': {feature['description']}\n"
        
        interpretation += (
            "\n\n**Recommendation:** Exercise caution. "
            "Verify the company through official channels."
        )
    else:
        interpretation = (
            f"This job posting appears **Legitimate** "
            f"with {confidence:.1f}% confidence.\n\n"
        )
        
        if legit_features:
            interpretation += "**Legitimate indicators:**\n"
            for feature in legit_features[:5]:
                interpretation += f"- '{feature['word']}': {feature['description']}\n"
        
        interpretation += (
            "\n\n**Recommendation:** While this appears legitimate, "
            "always verify the company."
        )
    
    # Combine all features for charts
    all_features = fraud_features + legit_features
    
    # Build chart data
    chart_words = [feature['word'] for feature in all_features[:10]]
    chart_weights = [feature['weight'] for feature in all_features[:10]]
    chart_colors = [
        '#FF6B6B' if feature['weight'] > 0 else '#4ECDC4'
        for feature in all_features[:10]
    ]
    
    return {
        'feature_contributions': all_features[:10],
        'positive_features': fraud_features[:5],
        'negative_features': legit_features[:5],
        'interpretation': interpretation,
        'highlighted_text': text,
        'chart_data': {
            'words': chart_words,
            'weights': chart_weights,
            'colors': chart_colors
        }
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns information about the server status and whether the model is loaded.
    Used by the frontend to verify the backend is running.
    
    Returns:
        JSON with status, model_loaded flag, and timestamp
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'timestamp': time.time()
    })


@app.route('/api/predict', methods=['POST'])
@handle_errors
def predict():
    """
    Make a prediction for a job posting.
    
    This is the main prediction endpoint. It accepts either:
        - 'text': A single text string with the full job posting
        - Individual fields: 'title', 'company_profile', 'description', 
          'requirements', 'benefits'
    
    Request Body (JSON):
        {
            "text": "Full job posting text..."
        }
        OR
        {
            "title": "Job Title",
            "description": "Job description...",
            ...
        }
    
    Returns:
        JSON with prediction label, confidence, and probabilities
    """
    # Get JSON data from request
    data = request.get_json()
    
    if not data:
        return jsonify({
            'success': False,
            'error': 'No data provided'
        }), 400
    
    # Get text from either 'text' field or combine individual fields
    text = data.get('text')
    
    if not text:
        # Combine individual fields if 'text' not provided
        fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        text = ' '.join(str(data.get(field, '')) for field in fields)
    
    # Validate text length
    if len(text.strip()) < 10:
        return jsonify({
            'success': False,
            'error': 'Text too short (minimum 10 characters)'
        }), 400
    
    # Record start time for timing
    start_time = time.time()
    
    # Clean the text and make prediction
    cleaned_text = clean_text(text)
    result = detector.predict(cleaned_text)
    
    # Calculate inference time
    inference_time_ms = round((time.time() - start_time) * 1000, 2)
    
    # Build response (model already returns percentages)
    response_data = {
        'success': True,
        'prediction': {
            'label': result['label'],
            'confidence': result['confidence'],
            'probabilities': {
                'legitimate': result['probabilities']['legitimate'],
                'fraudulent': result['probabilities']['fraudulent']
            },
            'is_fraudulent': result['prediction'] == 1
        },
        'metadata': {
            'text_length': len(text),
            'inference_time_ms': inference_time_ms
        }
    }
    
    # Add fraud signals if detected
    if 'fraud_signals' in result:
        response_data['prediction']['fraud_signals'] = result['fraud_signals']
    
    return jsonify(response_data)


@app.route('/api/explain', methods=['POST'])
@handle_errors
def explain():
    """
    Get a prediction with detailed explanation.
    
    Similar to /api/predict but also returns an explanation of
    why the model made its prediction, including keyword analysis.
    
    Request Body (JSON):
        {
            "text": "Full job posting text..."
        }
    
    Returns:
        JSON with prediction, explanation, and chart data
    """
    # Get JSON data from request
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'success': False,
            'error': 'No text provided'
        }), 400
    
    text = data['text']
    
    # Validate text length
    if len(text.strip()) < 10:
        return jsonify({
            'success': False,
            'error': 'Text too short (minimum 10 characters)'
        }), 400
    
    # Record start time
    start_time = time.time()
    
    # Make prediction
    cleaned_text = clean_text(text)
    prediction = detector.predict(cleaned_text)
    
    # Generate explanation
    explanation = generate_explanation(text, prediction)
    
    # Extract detailed posting analysis with comprehensive advisories
    detailed_analysis = detector.extract_posting_details(text)
    
    # Calculate processing time
    explanation_time_ms = round((time.time() - start_time) * 1000, 2)
    
    # Build response (model already returns percentages)
    response_data = {
        'success': True,
        'prediction': {
            'label': prediction['label'],
            'confidence': prediction['confidence'],
            'probabilities': {
                'legitimate': prediction['probabilities']['legitimate'],
                'fraudulent': prediction['probabilities']['fraudulent']
            },
            'is_fraudulent': prediction['prediction'] == 1
        },
        'explanation': explanation,
        'detailed_analysis': detailed_analysis,
        'metadata': {
            'text_length': len(text),
            'explanation_time_ms': explanation_time_ms
        }
    }
    
    # Add fraud signals if detected
    if 'fraud_signals' in prediction:
        response_data['prediction']['fraud_signals'] = prediction['fraud_signals']
    
    return jsonify(response_data)


@app.route('/api/batch-predict', methods=['POST'])
@handle_errors
def batch_predict():
    """
    Make predictions for multiple job postings at once.
    
    This endpoint is useful for bulk analysis of job postings.
    Maximum 100 texts per request.
    
    Request Body (JSON):
        {
            "texts": ["text1", "text2", "text3", ...]
        }
    
    Returns:
        JSON with list of predictions for each text
    """
    # Get JSON data from request
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({
            'success': False,
            'error': 'No texts provided'
        }), 400
    
    texts = data['texts']
    
    # Validate input
    if not isinstance(texts, list):
        return jsonify({
            'success': False,
            'error': 'texts must be an array'
        }), 400
    
    if len(texts) == 0:
        return jsonify({
            'success': False,
            'error': 'texts array is empty'
        }), 400
    
    if len(texts) > 100:
        return jsonify({
            'success': False,
            'error': 'Maximum 100 texts per request'
        }), 400
    
    # Process each text
    predictions = []
    
    for index, text in enumerate(texts):
        if len(text.strip()) < 10:
            # Text too short - add error entry
            predictions.append({
                'index': index,
                'error': 'Text too short'
            })
        else:
            # Make prediction
            cleaned_text = clean_text(text)
            result = detector.predict(cleaned_text)
            
            predictions.append({
                'index': index,
                'label': result['label'],
                'confidence': result['confidence'],  # Already a percentage from model
                'is_fraudulent': result['prediction'] == 1
            })
    
    return jsonify({
        'success': True,
        'predictions': predictions,
        'total': len(texts)
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Get information about the model.
    
    Returns:
        JSON with model name, type, and supported labels
    """
    return jsonify({
        'success': True,
        'model': {
            'name': 'Fake Job Detector',
            'type': 'BERT-based classifier',
            'labels': ['Legitimate', 'Fraudulent']
        }
    })


@app.route('/api/sample-jobs', methods=['GET'])
def get_sample_jobs():
    """
    Get sample job postings for testing.
    
    Returns example legitimate and suspicious job postings
    that can be used to test the system.
    
    Returns:
        JSON with sample legitimate and suspicious job texts
    """
    return jsonify({
        'success': True,
        'samples': {
            'legitimate': [{
                'title': 'Software Engineer',
                'description': (
                    "Bachelor's degree required. 3+ years experience. "
                    "Health insurance, 401k matching."
                )
            }],
            'suspicious': [{
                'title': 'EARN $5000/WEEK!!!',
                'description': (
                    'No experience needed! Send bank details! '
                    'Unlimited earning!'
                )
            }]
        }
    })


@app.route('/api/explain-image', methods=['POST'])
@handle_errors
def explain_image():
    """
    Get prediction with detailed explanation from an image.
    
    Extracts text from job posting image via OCR, then runs
    the same analysis pipeline as /api/explain.
    
    Request Body (JSON):
        {
            "image": "base64_encoded_image_data"
        }
    
    Returns:
        JSON with prediction, explanation, detailed analysis, and OCR metadata
    """
    import base64
    from io import BytesIO
    from PIL import Image
    from src.image_ocr import ImageOCR
    
    # Get JSON data from request
    data = request.get_json()
    
    if not data or 'image' not in data:
        return jsonify({
            'success': False,
            'error': 'No image provided'
        }), 400
    
    image_data = data['image']
    
    # Decode base64 image
    try:
        # Remove data URL prefix if present (e.g., "data:image/png;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Invalid image data: {str(e)}'
        }), 400
    
    # Record start time
    start_time = time.time()
    
    # Initialize OCR and extract text
    try:
        ocr = ImageOCR()
        extracted_text = ocr.extract_text(pil_image)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'OCR failed: {str(e)}'
        }), 500
    
    # Validate extracted text
    if not extracted_text or len(extracted_text.strip()) < 10:
        return jsonify({
            'success': False,
            'error': 'Could not extract enough text from image (minimum 10 characters)'
        }), 400
    
    # Use the same pipeline as /api/explain
    cleaned_text = clean_text(extracted_text)
    prediction = detector.predict(cleaned_text)
    explanation = generate_explanation(extracted_text, prediction)
    detailed_analysis = detector.extract_posting_details(extracted_text)
    
    # Calculate processing time
    processing_time_ms = round((time.time() - start_time) * 1000, 2)
    
    # Build response (same structure as /api/explain, with OCR metadata)
    response_data = {
        'success': True,
        'prediction': {
            'label': prediction['label'],
            'confidence': prediction['confidence'],
            'probabilities': {
                'legitimate': prediction['probabilities']['legitimate'],
                'fraudulent': prediction['probabilities']['fraudulent']
            },
            'is_fraudulent': prediction['prediction'] == 1
        },
        'explanation': explanation,
        'detailed_analysis': detailed_analysis,
        'metadata': {
            'source': 'image',
            'extracted_text': extracted_text,
            'text_length': len(extracted_text),
            'processing_time_ms': processing_time_ms
        }
    }
    
    # Add fraud signals if detected
    if 'fraud_signals' in prediction:
        response_data['prediction']['fraud_signals'] = prediction['fraud_signals']
    
    return jsonify(response_data)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Initialize the ML model
    initialize_services()
    
    # Start the Flask server
    app.run(
        host=FLASK_CONFIG["host"],
        port=FLASK_CONFIG["port"],
        debug=FLASK_CONFIG["debug"]
    )
