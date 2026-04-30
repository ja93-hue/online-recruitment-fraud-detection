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

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global variable to hold the fraud detector model
detector = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize input text before processing.

    Removes HTML tags, URLs, and extra whitespace.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def initialize_services():
    """
    Initialize the machine learning services.

    Creates a JobFraudDetector, loads the trained model if available,
    falls back to demo mode otherwise.
    """
    global detector

    logger.info("Initializing ML services...")
    detector = JobFraudDetector()
    model_path = MODEL_DIR / "best_model.pt"

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

    Catches any exceptions and returns a proper JSON error response.
    """
    @wraps(function)
    def decorated_function(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            logger.error(f"Error in {function.__name__}: {error}\n{traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': str(error)
            }), 500
    return decorated_function


# =============================================================================
# KEYWORD DICTIONARIES FOR EXPLANATIONS
# =============================================================================

# Keywords that indicate potential fraud
# Format: 'keyword': ('description', severity_score 1-5)
FRAUD_KEYWORDS = {
    'send ssn':        ('Requests SSN', 5),
    'bank details':    ('Requests banking info', 5),
    'credit card':     ('Requests credit card', 5),
    'wire money':      ('Requests money transfer', 5),
    'processing fee':  ('Upfront fees required', 4.5),
    'gift card':       ('Gift card scam indicator', 4.5),
    'earn money fast': ('Quick money promise', 4),
    '$5000':           ('Unrealistic salary claims', 4),
    'easy money':      ('Easy money claims', 3.5),
    'guaranteed':      ('Guarantee claims', 2.5),
    'no interview':    ('No interview required', 4),
    'no experience':   ('No experience required', 3),
    'urgent':          ('Urgency tactics', 2.5),
    '!!!':             ('Excessive punctuation', 2),
    'telegram':        ('Informal contact method', 2),
    'gmail.com':       ('Free email service', 2.5),
    'work from home':  ('WFH (can be misused)', 1.5),
}

# Keywords that indicate legitimate postings (negative weights)
LEGITIMATE_KEYWORDS = {
    'experience required': ('Experience requirements', 4),
    'bachelor':            ('Education requirements', 4),
    'degree':              ('Degree mentioned', 2.5),
    'qualifications':      ('Qualification requirements', 3),
    'health insurance':    ('Standard benefits', 4.5),
    '401k':                ('Retirement benefits', 4.5),
    'pto':                 ('Paid time off', 4),
    'competitive salary':  ('Professional language', 3),
    'interview':           ('Interview process', 3),
    'resume':              ('Resume required', 2),
}


def generate_explanation(text: str, prediction: dict) -> dict:
    """
    Generate a human-readable explanation of the prediction.

    Analyzes text for known keywords and generates an explanation of why
    the model made its prediction.

    Args:
        text:       The job posting text
        prediction: The model's prediction dictionary

    Returns:
        Dictionary with feature_contributions, positive_features,
        negative_features, interpretation, and chart_data.
    """
    text_lower = text.lower()
    fraud_features = []
    legit_features = []

    for keyword, (description, severity) in FRAUD_KEYWORDS.items():
        if keyword in text_lower:
            weight = round((0.1 + severity / 5 * 0.4), 3)
            fraud_features.append({
                'word': keyword,
                'weight': weight,
                'description': description
            })

    for keyword, (description, severity) in LEGITIMATE_KEYWORDS.items():
        if keyword in text_lower:
            weight = round(-(0.1 + severity / 5 * 0.4), 3)
            legit_features.append({
                'word': keyword,
                'weight': weight,
                'description': description
            })

    fraud_features.sort(key=lambda x: -x['weight'])
    legit_features.sort(key=lambda x: x['weight'])

    # FIX: Use final probabilities (already hybrid-adjusted) for interpretation
    final_fraud = prediction['probabilities']['fraudulent']
    confidence  = prediction['confidence']

    if final_fraud >= 70:
        interpretation = (
            f"This job posting is classified as **High Risk / Fraudulent** "
            f"with {confidence:.1f}% confidence.\n\n"
        )
        if fraud_features:
            interpretation += "**Suspicious indicators:**\n"
            for feature in fraud_features[:5]:
                interpretation += f"- '{feature['word']}': {feature['description']}\n"
        interpretation += (
            "\n\n**Recommendation:** Exercise extreme caution. "
            "Verify the company through official channels before sharing any information."
        )
    elif final_fraud >= 50:
        interpretation = (
            f"This job posting is classified as **Medium Risk** "
            f"with {confidence:.1f}% confidence.\n\n"
        )
        if fraud_features:
            interpretation += "**Suspicious indicators:**\n"
            for feature in fraud_features[:5]:
                interpretation += f"- '{feature['word']}': {feature['description']}\n"
        interpretation += (
            "\n\n**Recommendation:** Research this company carefully "
            "before sharing personal information."
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
            "always verify the company independently."
        )

    all_features  = fraud_features + legit_features
    chart_words   = [f['word']   for f in all_features[:10]]
    chart_weights = [f['weight'] for f in all_features[:10]]
    chart_colors  = [
        '#FF6B6B' if f['weight'] > 0 else '#4ECDC4'
        for f in all_features[:10]
    ]

    return {
        'feature_contributions': all_features[:10],
        'positive_features':     fraud_features[:5],
        'negative_features':     legit_features[:5],
        'interpretation':        interpretation,
        'highlighted_text':      text,
        'chart_data': {
            'words':   chart_words,
            'weights': chart_weights,
            'colors':  chart_colors
        }
    }


def build_prediction_response(result: dict) -> dict:
    """
    Build the standard prediction block returned by all endpoints.

    FIX: Separates bert_raw (raw BERT softmax) from final probabilities
    (which may be boosted by rule-based signals). This ensures the frontend
    can display two genuinely different scores.

    Args:
        result: Raw output from detector.predict()

    Returns:
        Dictionary with label, confidence, probabilities, bert_raw,
        is_fraudulent, and optional fraud_signals.
    """
    pred_block = {
        'label':      result['label'],
        'confidence': result['confidence'],
        'probabilities': {
            'legitimate': result['probabilities']['legitimate'],
            'fraudulent':  result['probabilities']['fraudulent']
        },
        # FIX: Store raw BERT scores separately so the frontend can show
        # the pure model score vs the hybrid (rule-boosted) final score.
        # If the model returns bert_raw, use it; otherwise mark as unavailable.
        'bert_raw': result.get('bert_raw', None),
        'is_fraudulent': result['prediction'] == 1
    }
    if 'fraud_signals' in result:
        pred_block['fraud_signals'] = result['fraud_signals']
    return pred_block


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns server status and whether the model is loaded.
    """
    return jsonify({
        'status':       'healthy',
        'model_loaded': detector is not None,
        'timestamp':    time.time()
    })


@app.route('/api/predict', methods=['POST'])
@handle_errors
def predict():
    """
    Make a prediction for a job posting.

    Accepts either a single 'text' field or individual fields
    (title, company_profile, description, requirements, benefits).

    Returns:
        JSON with prediction label, confidence, probabilities, bert_raw,
        and metadata.
    """
    data = request.get_json()

    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    text = data.get('text')
    if not text:
        fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        text   = ' '.join(str(data.get(f, '')) for f in fields)

    if len(text.strip()) < 10:
        return jsonify({'success': False, 'error': 'Text too short (minimum 10 characters)'}), 400

    start_time   = time.time()
    cleaned_text = clean_text(text)
    result       = detector.predict(cleaned_text)
    inference_ms = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        'success':    True,
        'prediction': build_prediction_response(result),
        'metadata': {
            'text_length':      len(text),
            'inference_time_ms': inference_ms
        }
    })


@app.route('/api/explain', methods=['POST'])
@handle_errors
def explain():
    """
    Get a prediction with detailed explanation.

    Same as /api/predict but also returns keyword analysis and chart data.

    Returns:
        JSON with prediction, explanation, detailed_analysis, and metadata.
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'No text provided'}), 400

    text = data['text']

    if len(text.strip()) < 10:
        return jsonify({'success': False, 'error': 'Text too short (minimum 10 characters)'}), 400

    start_time       = time.time()
    cleaned_text     = clean_text(text)
    prediction       = detector.predict(cleaned_text)
    explanation      = generate_explanation(text, prediction)
    detailed_analysis= detector.extract_posting_details(text)
    explanation_ms   = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        'success':         True,
        'prediction':      build_prediction_response(prediction),
        'explanation':     explanation,
        'detailed_analysis': detailed_analysis,
        'metadata': {
            'text_length':        len(text),
            'explanation_time_ms': explanation_ms
        }
    })


@app.route('/api/batch-predict', methods=['POST'])
@handle_errors
def batch_predict():
    """
    Make predictions for multiple job postings at once.

    Maximum 100 texts per request.

    Request Body:
        { "texts": ["text1", "text2", ...] }

    Returns:
        JSON with list of predictions for each text.
    """
    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({'success': False, 'error': 'No texts provided'}), 400

    texts = data['texts']

    if not isinstance(texts, list):
        return jsonify({'success': False, 'error': 'texts must be an array'}), 400
    if len(texts) == 0:
        return jsonify({'success': False, 'error': 'texts array is empty'}), 400
    if len(texts) > 100:
        return jsonify({'success': False, 'error': 'Maximum 100 texts per request'}), 400

    predictions = []
    for index, text in enumerate(texts):
        if len(text.strip()) < 10:
            predictions.append({'index': index, 'error': 'Text too short'})
        else:
            cleaned_text = clean_text(text)
            result       = detector.predict(cleaned_text)
            predictions.append({
                'index':        index,
                'label':        result['label'],
                'confidence':   result['confidence'],
                'is_fraudulent': result['prediction'] == 1,
                # FIX: include bert_raw in batch results too
                'bert_raw':     result.get('bert_raw', None)
            })

    return jsonify({
        'success':     True,
        'predictions': predictions,
        'total':       len(texts)
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.
    """
    return jsonify({
        'success': True,
        'model': {
            'name':   'Fake Job Detector',
            'type':   'BERT-based classifier',
            'labels': ['Legitimate', 'Fraudulent']
        }
    })


@app.route('/api/sample-jobs', methods=['GET'])
def get_sample_jobs():
    """
    Get sample job postings for testing.
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

    Extracts text via OCR, then runs the same pipeline as /api/explain.

    Request Body:
        { "image": "base64_encoded_image_data" }

    Returns:
        JSON with prediction, explanation, detailed_analysis, and OCR metadata.
    """
    import base64
    from io import BytesIO
    from PIL import Image
    from src.image_ocr import ImageOCR

    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    image_data = data['image']

    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        pil_image   = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return jsonify({'success': False, 'error': f'Invalid image data: {str(e)}'}), 400

    start_time = time.time()

    try:
        ocr            = ImageOCR()
        extracted_text = ocr.extract_text(pil_image)
    except Exception as e:
        return jsonify({'success': False, 'error': f'OCR failed: {str(e)}'}), 500

    if not extracted_text or len(extracted_text.strip()) < 10:
        return jsonify({
            'success': False,
            'error':   'Could not extract enough text from image (minimum 10 characters)'
        }), 400

    cleaned_text      = clean_text(extracted_text)
    prediction        = detector.predict(cleaned_text)
    explanation       = generate_explanation(extracted_text, prediction)
    detailed_analysis = detector.extract_posting_details(extracted_text)
    processing_ms     = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        'success':          True,
        'prediction':       build_prediction_response(prediction),
        'explanation':      explanation,
        'detailed_analysis': detailed_analysis,
        'metadata': {
            'source':          'image',
            'extracted_text':  extracted_text,
            'text_length':     len(extracted_text),
            'processing_time_ms': processing_ms
        }
    })


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    initialize_services()
    app.run(
        host=FLASK_CONFIG["host"],
        port=FLASK_CONFIG["port"],
        debug=FLASK_CONFIG["debug"]
    )