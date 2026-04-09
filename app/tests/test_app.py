"""
Unit Tests for Fake Job Detection Application
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import DataPreprocessor, TextPreprocessorForInference
from src.model import BertClassifier, JobFraudDetector
from src.explainer import LIMEExplainer


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def setup_method(self):
        self.preprocessor = DataPreprocessor()
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello World!"
        cleaned = self.preprocessor.clean_text(text)
        assert cleaned == "hello world!"
    
    def test_clean_text_html(self):
        """Test HTML tag removal."""
        text = "<p>Hello <b>World</b></p>"
        cleaned = self.preprocessor.clean_text(text)
        assert "<" not in cleaned
        assert ">" not in cleaned
    
    def test_clean_text_urls(self):
        """Test URL removal."""
        text = "Visit https://example.com for more info"
        cleaned = self.preprocessor.clean_text(text)
        assert "https" not in cleaned
        assert "example.com" not in cleaned
    
    def test_clean_text_email(self):
        """Test email removal."""
        text = "Contact us at test@example.com"
        cleaned = self.preprocessor.clean_text(text)
        assert "@" not in cleaned
    
    def test_clean_text_none(self):
        """Test handling of None input."""
        cleaned = self.preprocessor.clean_text(None)
        assert cleaned == ""
    
    def test_load_sample_data(self):
        """Test loading sample data."""
        df = self.preprocessor.load_dataset()
        assert len(df) > 0
        assert 'fraudulent' in df.columns
        assert 'title' in df.columns
    
    def test_preprocess_dataset(self):
        """Test full preprocessing pipeline."""
        df = self.preprocessor.load_dataset()
        df = self.preprocessor.preprocess_dataset(df)
        assert 'combined_text' in df.columns
        assert df['combined_text'].str.len().min() > 0


class TestTextPreprocessorForInference:
    """Tests for inference preprocessor."""
    
    def setup_method(self):
        self.preprocessor = TextPreprocessorForInference()
    
    def test_preprocess(self):
        """Test preprocessing for inference."""
        text = "Software Engineer at Google"
        cleaned = self.preprocessor.preprocess(text)
        assert cleaned == "software engineer at google"
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "Software Engineer"
        tokens = self.preprocessor.tokenize(text)
        assert 'input_ids' in tokens
        assert 'attention_mask' in tokens


class TestBertClassifier:
    """Tests for BERT classifier model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = BertClassifier()
        assert model is not None
        assert model.num_labels == 2


class TestJobFraudDetector:
    """Tests for JobFraudDetector class."""
    
    def setup_method(self):
        self.detector = JobFraudDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.model is None
        assert self.detector.tokenizer is not None
    
    def test_initialize_model(self):
        """Test model initialization."""
        self.detector.initialize_model()
        assert self.detector.model is not None
    
    def test_predict_without_model(self):
        """Test prediction fails without model."""
        with pytest.raises(ValueError):
            self.detector.predict("test text")


class TestLIMEExplainer:
    """Tests for LIME explainer."""
    
    def setup_method(self):
        self.explainer = LIMEExplainer()
    
    def test_initialization(self):
        """Test explainer initialization."""
        assert self.explainer.num_features == 10
        assert len(self.explainer.class_names) == 2
    
    def test_highlight_text(self):
        """Test text highlighting."""
        text = "test word example"
        contributions = [
            {'word': 'test', 'weight': 0.5},
            {'word': 'example', 'weight': -0.3}
        ]
        highlighted = self.explainer.highlight_text(text, contributions)
        assert '<span' in highlighted
    
    def test_get_chart_data(self):
        """Test chart data generation."""
        contributions = [
            {'word': 'test', 'weight': 0.5},
            {'word': 'example', 'weight': -0.3}
        ]
        chart_data = self.explainer.get_chart_data(contributions)
        assert 'words' in chart_data
        assert 'weights' in chart_data
        assert 'colors' in chart_data


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_sample_data(self):
        """Test full pipeline with sample data."""
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        df = preprocessor.load_dataset()
        df = preprocessor.preprocess_dataset(df)
        
        # Verify data
        assert len(df) > 0
        assert df['fraudulent'].nunique() == 2
        
        # Initialize detector
        detector = JobFraudDetector()
        detector.initialize_model()
        
        # Make prediction
        sample_text = df['combined_text'].iloc[0]
        result = detector.predict(sample_text)
        
        assert 'prediction' in result
        assert 'label' in result
        assert 'confidence' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
