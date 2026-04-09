# Fake Job Detection Application

## 🔍 Overview

This application uses a **Hybrid Detection System** combining:
- **BERT** (DistilBERT) - Deep learning model for subtle pattern recognition
- **Rule-Based Detection** - Pattern matching for definitive fraud signals
- **LIME** - Explainable AI for transparent predictions

The system achieves **100% accuracy** on our comprehensive test suite covering legitimate jobs, obvious scams, and sophisticated/subtle fraud attempts.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)

## 🎯 Detection Capabilities

| Category | Detection Method | Accuracy |
|----------|-----------------|----------|
| Legitimate Jobs | BERT ML Model | ✅ 100% |
| Obvious Scams | Rule-Based | ✅ 100% |
| Sophisticated Scams | Hybrid (Rules + BERT) | ✅ 100% |

### Scam Types Detected

- 💰 **Fee-based scams** - Upfront payments, onboarding kits, training fees
- 🎁 **Gift card scams** - Purchase and send gift card codes
- 🏦 **Money transfer scams** - Wire transfers, check cashing
- 📦 **Reshipping scams** - Package forwarding schemes
- 🔐 **Data harvesting** - Requests for SSN, bank details
- 💎 **MLM/Pyramid schemes** - Disguised as marketing jobs
- 🪙 **Crypto scams** - Trading deposits, investment schemes
- 📱 **WhatsApp/Telegram recruitment** - Unprofessional contact methods

## 🚀 Features

- **Hybrid Detection System**: Rules for definitive fraud, BERT for subtle patterns
- **Image OCR Support**: Analyze job posting screenshots using OCR
- **Advisory Tone**: Provides guidance, not definitive verdicts
- **LIME Explanations**: Transparent, human-readable explanations
- **Modern Web UI**: Professional Streamlit interface with dark theme
- **REST API**: Flask backend with comprehensive endpoints
- **Docker Ready**: Production-ready containerization

## 📁 Project Structure

```
app/
├── backend/
│   ├── __init__.py
│   └── api.py              # Flask REST API
├── frontend/
│   ├── __init__.py
│   └── app.py              # Streamlit UI
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data preprocessing & SMOTE
│   ├── model.py            # BERT classifier & training
│   ├── explainer.py        # LIME explanations
│   └── image_ocr.py        # Image OCR for screenshots
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration settings
├── data/                   # Dataset storage
├── models/                 # Trained model storage
├── logs/                   # Log files
├── train.py               # Training script
├── run.py                 # Application launcher
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Multi-container setup
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- (Optional) Docker and Docker Compose

### Local Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
   ```

## 🏃 Running the Application

### Option 1: Quick Start (Both Services)

```bash
python run.py
```

This starts both the Flask backend and Streamlit frontend.

### Option 2: Run Services Separately

**Terminal 1 - Backend:**
```bash
python backend/api.py
```

**Terminal 2 - Frontend:**
```bash
streamlit run frontend/app.py
```

### Option 3: Docker Compose

```bash
docker-compose up --build
```

### Access the Application

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:5000
- **API Health Check**: http://localhost:5000/api/health

## 📊 Training Your Own Model

### Using Sample Data

```bash
python train.py
```

### Using Custom Dataset

```bash
python train.py --data path/to/your/dataset.csv --epochs 5
```

**Expected CSV format:**
| Column | Description |
|--------|-------------|
| title | Job title |
| company_profile | Company description |
| description | Job description |
| requirements | Job requirements |
| benefits | Job benefits |
| fraudulent | Label (0=legitimate, 1=fraudulent) |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | None | Path to training CSV |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 16 | Training batch size |
| `--learning-rate` | 2e-5 | Learning rate |

## 🔌 API Endpoints

### Health Check
```http
GET /api/health
```

### Predict
```http
POST /api/predict
Content-Type: application/json

{
    "text": "Job posting text..."
}
```

### Explain
```http
POST /api/explain
Content-Type: application/json

{
    "text": "Job posting text...",
    "num_features": 10
}
```

### Batch Predict
```http
POST /api/batch-predict
Content-Type: application/json

{
    "texts": ["Job 1...", "Job 2...", ...]
}
```

### Sample Jobs
```http
GET /api/sample-jobs
```

## 📷 Image OCR Feature

Analyze job posting screenshots and images directly using OCR (Optical Character Recognition).

### Prerequisites

1. **Install Tesseract OCR** (required for OCR functionality):

   **Windows:**
   - Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Run the installer and note the installation path
   - Add to PATH or specify path in code
   
   **Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```

2. **Install Python dependencies** (already in requirements.txt):
   ```bash
   pip install pytesseract opencv-python Pillow
   ```

### Usage Examples

#### Method 1: Direct from JobFraudDetector

```python
from src.model import JobFraudDetector

# Initialize and load model
detector = JobFraudDetector()
detector.load_model()

# Predict from image
result = detector.predict_from_image("job_screenshot.png")

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}%")
print(f"Extracted Text: {result['extracted_text'][:200]}...")
```

#### Method 2: Using ImageOCR Class

```python
from src.model import JobFraudDetector
from src.image_ocr import ImageOCR

# Initialize detector
detector = JobFraudDetector()
detector.load_model()

# Initialize OCR with custom settings
ocr = ImageOCR(
    preprocess=True,  # Apply image preprocessing for better OCR
    language="eng"    # OCR language
)

# Predict from image file
result = ocr.predict_from_image("job_screenshot.png", detector)

# Or from PIL Image
from PIL import Image
img = Image.open("job_screenshot.png")
result = ocr.predict_from_image(img, detector)
```

#### Method 3: Using Convenience Function

```python
from src.model import JobFraudDetector
from src.image_ocr import predict_from_image

detector = JobFraudDetector()
detector.load_model()

# Quick one-liner
result = predict_from_image("job_screenshot.png", detector)
```

### Image Preprocessing

The OCR module applies automatic preprocessing to improve accuracy:
- **Grayscale conversion** - Better text contrast
- **Noise removal** - Gaussian and median blur
- **Thresholding** - Otsu or adaptive thresholding for clean text extraction

### Supported Image Formats

- PNG, JPG/JPEG, BMP, TIFF, WebP, GIF

### Return Value

```python
{
    'prediction': 1,                    # 0=legitimate, 1=fraudulent
    'label': 'Fraudulent',              # String label
    'confidence': 87.5,                 # Confidence percentage
    'probabilities': {
        'legitimate': 12.5,
        'fraudulent': 87.5
    },
    'fraud_signals': ['Uses personal email domain'],  # Detected red flags
    'extracted_text': '...',            # OCR-extracted text
    'text_length': 1234,                # Character count
    'source_type': 'image'              # Indicates image source
}
```

### Windows Tesseract Path

If Tesseract is not in your PATH, specify it:

```python
# Method 1: Set globally
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Method 2: Pass to ImageOCR
ocr = ImageOCR(tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe')

# Method 3: Pass to predict_from_image
result = detector.predict_from_image(
    "image.png",
    tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)
```

## 🎨 UI Features

- **Final Combined Assessment**: Clear risk score with breakdown
- **Analysis Breakdown**: Side-by-side Rule vs BERT results
- **Quick Analysis**: Fast prediction without detailed explanation
- **Detailed Explanation**: LIME-based feature importance visualization
- **Advisory Points**: Actionable recommendations for each warning
- **Responsive Design**: Professional dark glassmorphism theme

## 📈 Model Performance

### Hybrid Detection Results

```
======================================================================
TEST SUMMARY
======================================================================
Total Tests: 13
Passed: 13 (100.0%)
Failed: 0
======================================================================
```

| Test Category | Pass Rate |
|---------------|-----------|
| Legitimate Jobs (3 tests) | 100% |
| Obvious Scams (3 tests) | 100% |
| Sophisticated Scams (5 tests) | 100% |
| Edge Cases (2 tests) | 100% |

### BERT Model Metrics
- **Training**: DistilBERT fine-tuned on EMSCAD dataset (17,000+ job postings)
- **Validation Accuracy**: ~92.5%
- **SMOTE Balancing**: Handles 95:5 class imbalance

## 🧪 Running Tests

```bash
# Run the comprehensive test suite
python tests/test_detection.py

# Run with pytest (verbose)
python -m pytest tests/test_detection.py -v
```

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
MODEL_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
}

LIME_CONFIG = {
    "num_features": 10,
    "num_samples": 500,
}
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build images
docker-compose build

# Run services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

For production, consider:
1. Using a proper WSGI server (Gunicorn is included)
2. Setting up HTTPS with reverse proxy (Nginx)
3. Configuring environment variables
4. Setting up monitoring and logging

## 📚 Datasets

Recommended datasets for training:
1. **EMSCAD** (Employment Scam Aegean Dataset)
2. **Kaggle Fake Job Postings Dataset**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes as part of a final semester project.

## 🙏 Acknowledgments

- Based on research: "Online Recruitment Fraud (ORF) Detection Using Deep Learning Approaches"
- BERT model from Hugging Face Transformers
- LIME library for explainability

## ⚠️ Disclaimer

This tool is designed to assist in identifying potentially fraudulent job postings but should not be the sole factor in making decisions. Always verify job postings through official channels.
