"""
=============================================================================
Streamlit Frontend for Fake Job Detection
=============================================================================

This module provides the web user interface for the fake job detection system.
It uses Streamlit to create an interactive web application where users can:
    - Paste job posting text
    - Get quick predictions (legitimate or fraudulent)
    - View detailed explanations with charts

How to Run:
    streamlit run frontend/app.py

Requirements:
    - Backend API must be running on localhost:5000
    - See backend/api.py for starting the backend

Author: ORFD Project Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Streamlit - Web application framework
import streamlit as st

# Requests - For making HTTP calls to the backend API
import requests

# Re - For regex text cleaning
import re

# Plotly - For creating interactive charts
import plotly.graph_objects as go

# PIL - For image handling
try:
    from PIL import Image
    import io
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Job Posting Advisor",    # Browser tab title
    page_icon="🔍",                       # Browser tab icon
    layout="wide",                        # Use full width of browser
    initial_sidebar_state="collapsed"    # Hide sidebar by default
)

# Backend API URL - change this if backend runs on different host/port
API_URL = "http://localhost:5000"


# =============================================================================
# CSS STYLING
# =============================================================================

def load_css():
    """
    Load custom CSS styles for the application.
    
    This function injects CSS into the page to customize:
        - Fonts (using Inter from Google Fonts)
        - Background colors and gradients
        - Card designs with shadows
        - Button styles
        - Result card colors (green for legitimate, red for fraud)
    
    Note: Streamlit allows custom CSS via st.markdown with unsafe_allow_html=True
    """
    
    # Define CSS as a multi-line string
    css_styles = '''
    <style>
        /* Import Google Fonts - Inter is a clean, modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        /* Apply Inter font globally */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Fix broken Material Symbols icons - hide garbled text and show clean arrow */
        [data-testid="stIconMaterial"] {
            font-size: 0 !important;
            width: 24px !important;
            height: 24px !important;
            display: inline-flex !important;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        
        [data-testid="stIconMaterial"]::before {
            content: '▶';
            font-size: 12px !important;
            font-family: 'Inter', sans-serif !important;
            color: #94a3b8 !important;
            position: absolute;
        }
        
        /* Arrow down when expanded */
        details[open] [data-testid="stIconMaterial"]::before {
            content: '▼';
        }
        
        /* Main page background */
        .main {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
        }
        
        /* Dark theme for streamlit elements */
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        
        /* Container styling */
        .block-container {
            padding: 2rem 4rem !important;
            max-width: 1400px !important;
            margin: 0 auto !important;
        }
        
        /* Hero section - large and impressive */
        .hero-section {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
            padding: 3.5rem 3rem;
            border-radius: 24px;
            margin-bottom: 2.5rem;
            text-align: center;
            box-shadow: 0 25px 50px -12px rgba(59, 130, 246, 0.35);
            position: relative;
            overflow: hidden;
        }
        
        .hero-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            opacity: 0.5;
        }
        
        /* Hero title - big and bold */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            color: white;
            margin-bottom: 1rem;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            position: relative;
            letter-spacing: -0.02em;
        }
        
        /* Hero subtitle */
        .hero-subtitle {
            font-size: 1.3rem;
            color: rgba(255, 255, 255, 0.95);
            position: relative;
            font-weight: 500;
        }
        
        /* Section cards - glassmorphism style */
        .modern-card {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(20px);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .modern-card h3 {
            color: white !important;
            font-size: 1.3rem;
            font-weight: 600;
        }
        
        /* Text area - large and prominent */
        .stTextArea textarea {
            min-height: 320px !important;
            font-size: 1.05rem !important;
            line-height: 1.7 !important;
            padding: 1.5rem !important;
            border-radius: 16px !important;
            border: 2px solid rgba(59, 130, 246, 0.3) !important;
            background: rgba(15, 23, 42, 0.9) !important;
            color: #e2e8f0 !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextArea textarea::placeholder {
            color: #64748b !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2), 0 8px 25px rgba(59, 130, 246, 0.15) !important;
        }
        
        .stTextArea label {
            display: none !important;
        }
        
        /* Result cards - impressive with animations */
        .result-card-legitimate {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            border: none;
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 40px rgba(16, 185, 129, 0.3);
            animation: slideUp 0.5s ease-out;
        }
        
        .result-card-fraudulent {
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            border: none;
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 20px 40px rgba(239, 68, 68, 0.3);
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Result label */
        .result-label {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: white !important;
        }
        
        /* Buttons - large and attractive */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            padding: 1rem 2rem !important;
            border-radius: 12px !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.4) !important;
        }
        
        .stButton > button:active {
            transform: translateY(-1px) !important;
        }
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }
        
        .stButton > button:hover {
            opacity: 0.9;
        }
        
        /* Tags for displaying features */
        .feature-tag {
            display: inline-block;
            padding: 0.6rem 1.2rem;
            margin: 0.3rem;
            border-radius: 10px;
            font-size: 0.95rem;
            font-weight: 500;
            transition: transform 0.2s ease;
        }
        
        .feature-tag:hover {
            transform: scale(1.05);
        }
        
        /* Warning tag style */
        .tag-warning {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: #78350f;
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
        }
        
        /* Safe tag style */
        .tag-safe {
            background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
            color: #064e3b;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        /* Info box for explanations */
        .info-box {
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3b82f6;
            padding: 1.5rem;
            border-radius: 0 16px 16px 0;
            margin: 1.5rem 0;
            color: #e2e8f0;
            font-size: 1.05rem;
            line-height: 1.8;
        }
        
        /* Status indicator */
        .status-connected {
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
            font-size: 1rem;
            font-weight: 600;
            color: white;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }
        
        .status-disconnected {
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
            font-size: 1rem;
            font-weight: 600;
            color: white;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: white !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 1rem !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 12px;
            padding: 0.5rem;
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: #94a3b8 !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: white !important;
        }
        
        /* Caption styling */
        .stCaption {
            color: #64748b !important;
        }
        
        /* Warning message */
        .stAlert {
            background: rgba(251, 191, 36, 0.1) !important;
            border: 1px solid rgba(251, 191, 36, 0.3) !important;
            border-radius: 12px !important;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-color: #3b82f6 !important;
        }
        
        /* Hide Streamlit's default menu, footer, and header */
        #MainMenu, footer, header {
            visibility: hidden;
        }
        
        /* Expander styling - cleaner look */
        .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.7) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            color: #f1f5f9 !important;
            font-weight: 500 !important;
            padding: 0.75rem 1rem !important;
            font-size: 0.9rem !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(51, 65, 85, 0.8) !important;
            border-color: rgba(59, 130, 246, 0.4) !important;
        }
        
        .streamlit-expanderContent {
            background: rgba(15, 23, 42, 0.5) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
            padding: 1rem !important;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.5);
        }
        
        /* Footer styling */
        .footer-text {
            text-align: center;
            color: #64748b;
            padding: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 3rem;
            font-size: 0.95rem;
        }
        
        /* Placeholder box */
        .placeholder-box {
            text-align: center;
            color: #64748b;
            padding: 4rem 2rem;
            background: rgba(30, 41, 59, 0.5);
            border-radius: 20px;
            border: 2px dashed rgba(100, 116, 139, 0.3);
        }
        
        .placeholder-icon {
            font-size: 4rem;
            margin-bottom: 1.5rem;
            opacity: 0.7;
        }
        
        .placeholder-text {
            font-size: 1.2rem;
            font-weight: 500;
        }
        
        /* Better column gap for side-by-side layout */
        [data-testid="column"] {
            padding: 0 0.5rem;
        }
        
        /* Ensure readable text everywhere */
        p, div, span, li {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* GLOBAL TEXT VISIBILITY FIXES */
        /* Make all text white/light by default */
        .stApp, .main, .block-container {
            color: #f1f5f9 !important;
        }
        
        /* All paragraph and text elements */
        p, span, div, label, h1, h2, h3, h4, h5, h6 {
            color: #f1f5f9 !important;
        }
        
        /* Markdown text */
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color: #f1f5f9 !important;
        }
        
        /* Streamlit text elements */
        .stText, [data-testid="stText"] {
            color: #f1f5f9 !important;
        }
        
        /* Success/Warning/Info/Error messages */
        .stSuccess, [data-testid="stSuccess"] {
            background-color: rgba(34, 197, 94, 0.15) !important;
            color: #4ade80 !important;
        }
        .stSuccess p, .stSuccess span {
            color: #4ade80 !important;
        }
        
        .stWarning, [data-testid="stWarning"] {
            background-color: rgba(251, 191, 36, 0.15) !important;
            color: #fbbf24 !important;
        }
        .stWarning p, .stWarning span {
            color: #fbbf24 !important;
        }
        
        .stInfo, [data-testid="stInfo"] {
            background-color: rgba(59, 130, 246, 0.15) !important;
            color: #60a5fa !important;
        }
        .stInfo p, .stInfo span {
            color: #60a5fa !important;
        }
        
        .stError, [data-testid="stError"] {
            background-color: rgba(239, 68, 68, 0.15) !important;
            color: #f87171 !important;
        }
        .stError p, .stError span {
            color: #f87171 !important;
        }
        
        /* Expander content text */
        .streamlit-expanderContent p, 
        .streamlit-expanderContent span,
        .streamlit-expanderContent div {
            color: #f1f5f9 !important;
        }
        
        /* Caption text - make it visible but dimmer */
        .stCaption, [data-testid="stCaption"], small {
            color: #94a3b8 !important;
        }
        
        /* Metric labels and values */
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
        }
        [data-testid="stMetricValue"] {
            color: #f1f5f9 !important;
        }
        
        /* File uploader styling - dark theme compatible */
        [data-testid="stFileUploader"] {
            background-color: #1e293b !important;
            border: 2px dashed #60a5fa !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
        }
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] section {
            color: #e2e8f0 !important;
        }
        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] {
            background-color: #1e293b !important;
            border: 2px dashed #60a5fa !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] p {
            color: #e2e8f0 !important;
        }
        /* Upload icon visibility */
        [data-testid="stFileUploaderDropzone"] svg {
            fill: #60a5fa !important;
            stroke: #60a5fa !important;
        }
        /* Browse files button */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
        }
        
        /* Tab content */
        .stTabs [data-baseweb="tab-panel"] {
            color: #f1f5f9 !important;
        }
        
        /* Bold text */
        strong, b {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Links */
        a {
            color: #60a5fa !important;
        }
        
        /* Lists */
        ul, ol, li {
            color: #f1f5f9 !important;
        }
    </style>
    '''
    
    # Inject the CSS into the page
    st.markdown(css_styles, unsafe_allow_html=True)


# =============================================================================
# API COMMUNICATION FUNCTIONS
# =============================================================================

def check_api() -> bool:
    """
    Check if the backend API is running and accessible.
    
    Makes a simple GET request to the /api/health endpoint.
    
    Returns:
        True if API is accessible, False otherwise
    """
    try:
        response = requests.get(
            f"{API_URL}/api/health",
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


def predict_job(text: str) -> dict:
    """
    Send job posting text to the API for quick prediction.
    
    Args:
        text: The job posting text to analyze
    
    Returns:
        Dictionary with prediction results, or None if error
    
    Example Response:
        {
            'success': True,
            'prediction': {
                'label': 'Legitimate',
                'confidence': 95.5,
                'is_fraudulent': False
            }
        }
    """
    try:
        response = requests.post(
            f"{API_URL}/api/predict",
            json={"text": text},
            timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as error:
        st.error(f"Error connecting to API: {error}")
        return None


def batch_predict(texts: list) -> dict:
    """
    Send multiple job postings to the API for batch prediction.
    
    Args:
        texts: List of job posting texts to analyze
    
    Returns:
        Dictionary with batch prediction results
    """
    try:
        response = requests.post(
            f"{API_URL}/api/batch-predict",
            json={"texts": texts},
            timeout=120  # 2 minute timeout for batch
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as error:
        st.error(f"Error connecting to API: {error}")
        return None


def get_explanation(text: str) -> dict:
    """
    Send job posting text to the API for detailed analysis with explanation.
    
    Similar to predict_job but returns additional explanation data
    including keyword analysis and charts.
    
    Args:
        text: The job posting text to analyze
    
    Returns:
        Dictionary with prediction and explanation, or None if error
    """
    try:
        response = requests.post(
            f"{API_URL}/api/explain",
            json={"text": text},
            timeout=120  # 2 minute timeout (explanation takes longer)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as error:
        st.error(f"Error connecting to API: {error}")
        return None


def explain_image(image_data: bytes) -> dict:
    """
    Send job posting image to the API for OCR + detailed analysis.
    
    Extracts text from image via OCR, then returns the same
    detailed analysis as get_explanation().
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        Dictionary with prediction, explanation, and extracted text, or None if error
    """
    import base64
    
    try:
        # Convert image bytes to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        response = requests.post(
            f"{API_URL}/api/explain-image",
            json={"image": image_b64},
            timeout=180  # 3 minute timeout (OCR + explanation)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json() if response.content else {}
            st.error(f"Error: {error_data.get('error', 'Unknown error')}")
            return None
            
    except Exception as error:
        st.error(f"Error connecting to API: {error}")
        return None


# =============================================================================
# UI RENDERING FUNCTIONS
# =============================================================================

def render_result(result: dict):
    """
    Render the prediction result card with advisory messaging.
    
    Shows clear differentiation between:
    1. Final Combined Assessment (at top)
    2. Rule-Based Detection (definitive fraud signals)
    3. BERT AI Analysis (subtle pattern detection)
    
    Args:
        result: Dictionary containing the API response with prediction
    """
    # Extract prediction data from result
    prediction = result.get('prediction', {})
    is_fraud = prediction.get('is_fraudulent', False)
    confidence = prediction.get('confidence', 0)
    fraud_signals = prediction.get('fraud_signals', [])
    
    # Get both hybrid (final) and BERT raw probabilities
    probabilities = prediction.get('probabilities', {})
    bert_raw = prediction.get('bert_raw', probabilities)  # Fallback to final if no raw
    
    final_fraud = probabilities.get('fraudulent', 0)
    final_legit = probabilities.get('legitimate', 0)
    bert_fraud = bert_raw.get('fraudulent', final_fraud)
    bert_legit = bert_raw.get('legitimate', final_legit)
    
    # Determine if rules contributed to the score
    rules_triggered = len(fraud_signals) > 0
    
    # ============================================
    # SECTION 1: FINAL COMBINED ASSESSMENT
    # ============================================
    st.markdown("### 🎯 Final Combined Assessment")
    
    # Choose styling based on final risk
    if is_fraud:
        card_class = "result-card-fraudulent"
        icon = "⚠️"
        label = "High Risk - Exercise Caution"
        sublabel = "We recommend verifying this posting carefully"
    else:
        card_class = "result-card-legitimate"
        icon = "✓"
        label = "Lower Risk - Appears Legitimate"
        sublabel = "Always verify company details independently"
    
    # Render the FINAL result card with advisory language
    result_html = f'''
    <div class="{card_class}">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="result-label">{label}</div>
        <div style="font-weight: 500; color: rgba(255,255,255,0.9); font-size: 1rem; margin-top: 0.5rem;">{sublabel}</div>
        <div style="font-weight: 700; color: white; font-size: 1.4rem; margin-top: 1rem;">Final Risk Score: {final_fraud}%</div>
        <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-top: 0.5rem;">
            Combined from Rule Detection + BERT AI Analysis
        </div>
    </div>
    '''
    st.markdown(result_html, unsafe_allow_html=True)
    
    # ============================================
    # SECTION 2: ANALYSIS BREAKDOWN
    # ============================================
    st.markdown("---")
    st.markdown("### 📊 Analysis Breakdown")
    
    col_rules, col_bert = st.columns(2)
    
    # --- Rule-Based Detection Column ---
    with col_rules:
        rule_color = "#ef4444" if rules_triggered else "#22c55e"
        rule_icon = "🚨" if rules_triggered else "✓"
        rule_status = "Detected" if rules_triggered else "Clear"
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.8); border: 2px solid {rule_color}; border-radius: 12px; padding: 1rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{rule_icon}</div>
            <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500;">Rule-Based Detection</div>
            <div style="color: {rule_color}; font-size: 1.3rem; font-weight: 700; margin-top: 0.5rem;">{rule_status}</div>
            <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">{len(fraud_signals)} warning sign(s) found</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("*Checks for: fees, personal emails, gift cards, urgency tactics, etc.*")
    
    # --- BERT AI Analysis Column ---
    with col_bert:
        bert_is_risky = bert_fraud > 50
        bert_color = "#ef4444" if bert_is_risky else "#22c55e"
        bert_icon = "🤖" 
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.8); border: 2px solid {bert_color}; border-radius: 12px; padding: 1rem; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{bert_icon}</div>
            <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500;">BERT AI Analysis</div>
            <div style="color: {bert_color}; font-size: 1.3rem; font-weight: 700; margin-top: 0.5rem;">{bert_fraud}% Risk</div>
            <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">Pattern-based ML detection</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("*BERT catches subtle scam patterns learned from training data*")
    
    # ============================================
    # SECTION 3: FRAUD SIGNALS (if any)
    # ============================================
    if fraud_signals:
        st.markdown("---")
        render_fraud_signals_advisory(fraud_signals)


def render_fraud_signals_advisory(signals: list):
    """
    Render fraud signals with advisory recommendations.
    
    Each signal includes a helpful recommendation for the user
    to take action, rather than a definitive verdict.
    
    Args:
        signals: List of detected fraud signal strings
    """
    # Map signals to advisory recommendations
    advisory_map = {
        "Requests upfront payment/fees": "💡 Legitimate employers typically don't ask applicants to pay fees. We recommend confirming this with the company directly.",
        "Uses personal email domain": "💡 This posting uses a personal email (gmail/yahoo/etc). We suggest verifying the company's official contact through their website.",
        "Mentions gift cards": "💡 Gift card requests are a common scam tactic. We strongly recommend avoiding any job requiring gift card purchases.",
        "Involves money transfers": "💡 Jobs involving money transfers may be money laundering schemes. Please research this company thoroughly.",
        "Involves receiving checks": "💡 Check cashing schemes are common scams. We advise verifying this opportunity with consumer protection agencies.",
        "No experience/interview required": "💡 Most legitimate jobs require interviews. Consider whether this offer seems realistic.",
        "Uses urgency tactics": "💡 Pressure tactics like 'ACT NOW' are common in scams. Take your time to research before applying.",
        "Unrealistic salary claims": "💡 This salary seems unusually high. We recommend checking typical pay rates on sites like Glassdoor.",
        "Requests sensitive personal info": "💡 Be cautious about sharing SSN or bank details before formal hiring. Verify the employer first.",
        "Excessive caps/exclamation marks": "💡 Professional job postings typically use standard formatting. This style is common in scam posts."
    }
    
    st.markdown("### ⚠️ Points to Consider")
    
    for signal in signals:
        advisory = advisory_map.get(signal, f"💡 Please verify this aspect of the posting independently.")
        
        signal_html = f'''
        <div style="background: rgba(251, 191, 36, 0.15); border-left: 4px solid #fbbf24; 
                    padding: 1rem 1.25rem; border-radius: 0 12px 12px 0; margin: 0.75rem 0;">
            <div style="color: #fcd34d; font-weight: 600; margin-bottom: 0.5rem;">🚩 {signal}</div>
            <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.6;">{advisory}</div>
        </div>
        '''
        st.markdown(signal_html, unsafe_allow_html=True)


def render_explanation(data: dict):
    """
    Render the detailed explanation in a clean, simple format.
    """
    explanation = data.get('explanation', {})
    detailed_analysis = data.get('detailed_analysis', {})
    prediction = data.get('prediction', {})
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📋 Advisory", "📊 Key Factors", "📝 Summary"])
    
    # === TAB 1: Advisory ===
    with tab1:
        # Risk banner
        risk_summary = detailed_analysis.get('risk_summary', {})
        overall_risk = risk_summary.get('overall_risk', 'UNKNOWN')
        overall_advice = risk_summary.get('overall_advice', 'Analysis complete.')
        
        risk_colors = {'CRITICAL': '#dc2626', 'HIGH': '#ea580c', 'MEDIUM': '#ca8a04', 'LOW': '#16a34a'}
        risk_icons = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
        color = risk_colors.get(overall_risk, '#6b7280')
        icon = risk_icons.get(overall_risk, '⚪')
        
        st.markdown(f"""
        <div style="background: {color}20; border: 2px solid {color}; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <div style="font-size: 20px; font-weight: bold; color: {color}; margin-bottom: 8px;">
                {icon} Risk Level: {overall_risk}
            </div>
            <div style="color: #e2e8f0; font-size: 14px; line-height: 1.6;">{overall_advice}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Two columns: Findings | Details
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.markdown("**🔍 Risk Findings**")
            advisories = detailed_analysis.get('detailed_advisories', [])
            if advisories:
                for adv in advisories:
                    cat = adv.get('category', 'Finding')
                    find = adv.get('finding', '')
                    level = adv.get('risk_level', 'Info')
                    advice = adv.get('advisory', '')
                    adv_icon = adv.get('icon', '📌')
                    
                    level_color = {'Critical': '#ef4444', 'High': '#f97316', 'Medium': '#eab308', 'Low': '#22c55e'}.get(level, '#3b82f6')
                    
                    with st.expander(f"{adv_icon} {cat}: {find}"):
                        st.markdown(f"""
                        <span style="background: {level_color}; color: white; padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">{level}</span>
                        <div style="margin-top: 12px; color: #e2e8f0; line-height: 1.7; font-size: 14px;">{advice}</div>
                        """, unsafe_allow_html=True)
            else:
                st.success("No specific red flags detected.")
        
        with right_col:
            st.markdown("**📑 Extracted Info**")
            info = detailed_analysis.get('extracted_info', {})
            
            # Display extracted details simply
            if info.get('emails'):
                emails = info['emails']
                is_personal = any('@gmail' in e or '@yahoo' in e or '@hotmail' in e for e in emails)
                st.markdown(f"📧 **Email:** <span style='color: {'#ef4444' if is_personal else '#22c55e'}'>{', '.join(emails)}</span>", unsafe_allow_html=True)
            
            if info.get('phone_numbers'):
                st.markdown(f"📞 **Phone:** {', '.join(info['phone_numbers'])}")
            
            if info.get('company_mentions'):
                st.markdown(f"🏢 **Company:** {', '.join(info['company_mentions'][:2])}")
            
            if info.get('work_arrangement'):
                st.markdown(f"💼 **Work Type:** {info['work_arrangement'].replace('_', ' ').title()}")
            
            if info.get('messaging_apps'):
                # messaging_apps is a boolean flag, not a list
                st.markdown(f"💬 **Messaging:** <span style='color: #f97316'>WhatsApp/Telegram detected ⚠️</span>", unsafe_allow_html=True)
            
            if info.get('requests_sensitive_info'):
                # requests_sensitive_info is a boolean flag
                st.markdown(f"🔐 **Info Requested:** <span style='color: #ef4444'>SSN/Bank details requested ⚠️</span>", unsafe_allow_html=True)
            
            if not any([info.get('emails'), info.get('phone_numbers'), info.get('company_mentions')]):
                st.info("No contact details extracted.")
    
    # === TAB 2: Key Factors ===
    with tab2:
        chart_data = explanation.get('chart_data', {})
        words = chart_data.get('words', []) if chart_data else []
        weights = chart_data.get('weights', []) if chart_data else []
        
        if words and weights:
            try:
                colors = ['#ef4444' if w > 0 else '#22c55e' for w in weights]
                fig = go.Figure(go.Bar(x=weights, y=words, orientation='h', marker=dict(color=colors)))
                fig.update_layout(
                    title=dict(text="Word Impact on Prediction", font=dict(color='#ffffff', size=18)),
                    xaxis_title="Impact (+ = risk, - = safe)", 
                    xaxis=dict(title_font=dict(color='#e2e8f0'), tickfont=dict(color='#e2e8f0')),
                    yaxis=dict(tickfont=dict(color='#e2e8f0')),
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Chart could not be rendered.")
        else:
            # Fallback: show fraud signals as key factors
            fraud_signals = prediction.get('fraud_signals', [])
            if fraud_signals:
                st.markdown("**Key Risk Indicators Found:**")
                for signal in fraud_signals:
                    st.markdown(f"🚩 {signal}")
            else:
                st.info("No key word factors available. See the Advisory tab for detailed analysis.")
    
    # === TAB 3: Summary ===
    with tab3:
        interpretation = explanation.get('interpretation', '')
        
        # Get final and BERT raw scores for the summary
        probs = prediction.get('probabilities', {})
        bert_raw = prediction.get('bert_raw', probs)
        final_risk = probs.get('fraudulent', 0)
        bert_risk = bert_raw.get('fraudulent', final_risk)
        fraud_signals = prediction.get('fraud_signals', [])
        
        # ===== FINAL SCORE SUMMARY BOX =====
        is_high_risk = final_risk > 50
        summary_color = "#dc2626" if is_high_risk else "#16a34a"
        summary_icon = "⚠️" if is_high_risk else "✓"
        summary_label = "HIGH RISK" if is_high_risk else "LOW RISK"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {summary_color}22 0%, {summary_color}11 100%); 
                    border: 2px solid {summary_color}; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{summary_icon}</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {summary_color}; margin-bottom: 0.5rem;">
                {summary_label}
            </div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.75rem;">
                Final Score: {final_risk}%
            </div>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.85rem;">Rule Detection</div>
                    <div style="color: {'#ef4444' if len(fraud_signals) > 0 else '#22c55e'}; font-size: 1.1rem; font-weight: 600;">
                        {len(fraud_signals)} warning(s)
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.85rem;">BERT Analysis</div>
                    <div style="color: {'#ef4444' if bert_risk > 50 else '#22c55e'}; font-size: 1.1rem; font-weight: 600;">
                        {bert_risk}% risk
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show interpretation or create one from risk summary
        if interpretation:
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 15px; border-radius: 0 8px 8px 0; color: #e2e8f0; line-height: 1.7;">
                {interpretation}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 15px; border-radius: 0 8px 8px 0; color: #e2e8f0; line-height: 1.7;">
                <strong>Risk Assessment: {overall_risk}</strong><br><br>
                {overall_advice}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Two columns for indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⚠️ Suspicious Indicators**")
            pos_features = explanation.get('positive_features', [])
            fraud_signals = prediction.get('fraud_signals', [])
            
            if pos_features:
                for f in pos_features[:5]:
                    word = f.get('word', str(f)) if isinstance(f, dict) else str(f)
                    st.warning(word)
            elif fraud_signals:
                for sig in fraud_signals[:5]:
                    short = sig[:30] + "..." if len(sig) > 30 else sig
                    st.warning(short)
            else:
                st.caption("None identified")
        
        with col2:
            st.markdown("**✓ Legitimate Indicators**")
            neg_features = explanation.get('negative_features', [])
            
            if neg_features:
                for f in neg_features[:5]:
                    word = f.get('word', str(f)) if isinstance(f, dict) else str(f)
                    st.success(word)
            else:
                st.caption("None identified")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main function that runs the Streamlit application.
    
    This function:
        1. Loads CSS styles
        2. Renders the hero section
        3. Checks API connection status
        4. Creates two-column layout: input on left, results on right
        5. Handles user interactions using session state
    """
    # Load custom CSS styles
    load_css()
    
    # Initialize session state for storing results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = None
    
    # === HERO SECTION ===
    hero_html = '''
    <div class="hero-section">
        <h1 class="hero-title">Job Posting Advisor</h1>
        <p class="hero-subtitle">AI-powered risk assessment to help you identify potential scams</p>
    </div>
    '''
    st.markdown(hero_html, unsafe_allow_html=True)
    
    # === API STATUS INDICATOR ===
    is_connected = check_api()
    
    if is_connected:
        status_text = "✅ API Connected"
        status_class = "status-connected"
    else:
        status_text = "❌ API Disconnected - Click Refresh or restart backend"
        status_class = "status-disconnected"
    
    # Status only (no refresh button)
    st.markdown(f'''<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <div class="{status_class}" style="flex: 1;">{status_text}</div>
    </div>''', unsafe_allow_html=True)
    
    # === MAIN LAYOUT: Two columns side by side ===
    col_input, col_results = st.columns(2, gap="large")
    
    # === LEFT COLUMN: Input Section ===
    with col_input:
        st.markdown(
            '<div class="modern-card"><h3 style="margin:0;">Enter Job Posting</h3></div>',
            unsafe_allow_html=True
        )
        
        # Text input area
        job_text = st.text_area(
            label="Paste job posting:",
            height=280,
            placeholder="Paste the complete job posting here...",
            label_visibility="collapsed",
            key="job_input"
        )
        
        # Character count
        st.caption(f"Characters: {len(job_text)}")
        
        # Analysis buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            quick_clicked = st.button(
                "Quick Analysis",
                use_container_width=True,
                disabled=not is_connected,
                key="quick_btn"
            )
        
        with col_btn2:
            detailed_clicked = st.button(
                "Detailed Analysis",
                use_container_width=True,
                disabled=not is_connected,
                key="detailed_btn"
            )
        
        # Handle button clicks
        if quick_clicked and job_text:
            with st.spinner("Analyzing..."):
                result = predict_job(job_text)
                if result and result.get('success'):
                    st.session_state.analysis_result = result
                    st.session_state.analysis_type = 'quick'
        
        if detailed_clicked and job_text:
            with st.spinner("Generating detailed analysis..."):
                result = get_explanation(job_text)
                if result and result.get('success'):
                    st.session_state.analysis_result = result
                    st.session_state.analysis_type = 'detailed'
        
        if not job_text and (quick_clicked or detailed_clicked):
            st.warning("Please enter a job posting to analyze.")
        
        # === BATCH MODE ===
        with st.expander("📦 Batch Mode - Analyze Multiple Postings"):
            st.markdown("""<p style='color: #94a3b8; font-size: 0.9rem; margin: 0 0 1rem 0;'>
                <strong>How to use:</strong> Enter multiple job postings in the text area above.<br>
                Separate each posting with <code style='background: rgba(59,130,246,0.2); padding: 2px 6px; border-radius: 4px; color: #60a5fa;'>---</code> on its own line.
            </p>""", unsafe_allow_html=True)
            batch_clicked = st.button(
                "Batch Analyze",
                use_container_width=True,
                disabled=not is_connected,
                key="batch_btn"
            )
            
            if batch_clicked and job_text:
                # Split by --- separator
                postings = [p.strip() for p in job_text.split('---') if p.strip() and len(p.strip()) > 10]
                
                if len(postings) == 0:
                    st.warning("No valid postings found. Make sure text is longer than 10 characters.")
                elif len(postings) == 1:
                    st.info("Only 1 posting detected. Use Quick Analysis for single postings.")
                elif len(postings) > 100:
                    st.error(f"Too many postings ({len(postings)}). Maximum is 100.")
                else:
                    with st.spinner(f"Analyzing {len(postings)} postings..."):
                        result = batch_predict(postings)
                        if result and result.get('success'):
                            st.success(f"✅ Analyzed {result.get('total', len(postings))} postings")
                            for pred in result.get('predictions', []):
                                idx = pred.get('index', 0) + 1
                                label = pred.get('label', 'Unknown')
                                conf = pred.get('confidence', 0)
                                icon = "✅" if label == "Legitimate" else "⚠️"
                                color = "#10b981" if label == "Legitimate" else "#ef4444"
                                st.markdown(f"**{icon} Posting {idx}:** <span style='color:{color}'>{label}</span> ({conf}%)", unsafe_allow_html=True)
                        else:
                            st.error("Batch analysis failed. Check API connection.")
            elif batch_clicked and not job_text:
                st.warning("Please enter job postings separated by ---")
        
        # === IMAGE OCR MODE ===
        if IMAGE_SUPPORT:
            st.markdown("""<p style='color: #94a3b8; font-size: 0.9rem; margin: 0 0 1rem 0;'>
                <strong>How to use:</strong> Upload a screenshot of a suspicious job posting.<br>
                Supported formats: <code style='background: rgba(59,130,246,0.2); padding: 2px 6px; border-radius: 4px; color: #60a5fa;'>PNG, JPG, JPEG</code>
            </p>""", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload image (PNG, JPG, JPEG)",
                type=['png', 'jpg', 'jpeg'],
                key="ocr_upload",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")
                
                # Analysis buttons for image
                col_img1, col_img2 = st.columns(2)
                
                with col_img1:
                    img_quick = st.button("Quick Analysis", use_container_width=True, disabled=not is_connected, key="img_quick_btn")
                
                with col_img2:
                    img_detailed = st.button("Detailed Analysis", use_container_width=True, disabled=not is_connected, key="img_detailed_btn")
                
                if img_quick or img_detailed:
                    # Get image bytes
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    with st.spinner("Extracting text and analyzing..." if img_detailed else "Extracting text..."):
                        result = explain_image(image_bytes)
                        if result and result.get('success'):
                            # Show extracted text info
                            metadata = result.get('metadata', {})
                            text_len = metadata.get('text_length', 0)
                            extracted_text = metadata.get('extracted_text', '')
                            st.success(f"✅ Extracted {text_len} characters from image")
                            st.text_area("Extracted Text", extracted_text, height=150, disabled=True, key="extracted_text_display")
                            # Store result - use appropriate type based on button clicked
                            st.session_state.analysis_result = result
                            st.session_state.analysis_type = 'detailed' if img_detailed else 'quick'
                            st.rerun()
                        elif result:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                        else:
                            st.error("Analysis failed. Check API connection.")
    # === RIGHT COLUMN: Results Section ===
    with col_results:
        st.markdown(
            '<div class="modern-card"><h3 style="margin:0;">Analysis Results</h3></div>',
            unsafe_allow_html=True
        )
        
        # Display results from session state
        if st.session_state.analysis_result:
            render_result(st.session_state.analysis_result)
            
            if st.session_state.analysis_type == 'detailed':
                render_explanation(st.session_state.analysis_result)
        else:
            placeholder_html = '''
            <div class="placeholder-box">
                <div class="placeholder-icon">🔍</div>
                <p class="placeholder-text">Paste a job posting to get risk assessment guidance</p>
            </div>
            '''
            st.markdown(placeholder_html, unsafe_allow_html=True)
    
    # === DISCLAIMER ===
    disclaimer_html = '''
    <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); 
                border-radius: 12px; padding: 1.25rem; margin-top: 2rem; text-align: center;">
        <div style="color: #93c5fd; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Important Disclaimer</div>
        <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
            This tool provides risk assessment guidance only, not definitive fraud detection. 
            Always verify job postings independently through official company websites, 
            LinkedIn profiles, and trusted review sites before sharing personal information.
        </div>
    </div>
    '''
    st.markdown(disclaimer_html, unsafe_allow_html=True)
    
    # === FOOTER ===
    st.markdown(
        '<div class="footer-text">Your data is processed locally | Built with BERT & Streamlit | For educational purposes</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
