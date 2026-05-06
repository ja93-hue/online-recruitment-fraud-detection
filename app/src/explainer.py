"""
=============================================================================
LIME Explainability Module for Fake Job Detection
=============================================================================

This module provides explanations for model predictions using LIME
(Local Interpretable Model-agnostic Explanations).

LIME works by:
    1. Creating variations of the input text
    2. Getting predictions for each variation
    3. Identifying which words most influenced the prediction

This helps users understand WHY the model made a certain prediction,
not just what the prediction is.

Author: ORFD Project Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard libraries
import numpy as np
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Callable, Optional

# LIME library for explanations
from lime.lime_text import LimeTextExplainer

# Add parent directory for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import configuration
from config.settings import LIME_CONFIG, LABEL_MAP

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# LIME EXPLAINER CLASS
# =============================================================================

class LIMEExplainer:
    # Class docstring removed for error-free state
    def _merge_phrase_features(self, text: str, word_weights: list) -> list:

        # Define phrases to merge and their priorities
        phrase_map = {
            'no interview required': ['no', 'interview', 'required'],
            'no interview needed': ['no', 'interview', 'needed'],
            'no interview': ['no', 'interview'],
            'interview required': ['interview', 'required'],
            'interview needed': ['interview', 'needed'],
        }
        # Lowercase text for matching
        text_lc = text.lower()
        # Build a word-to-weight dict
        word_weight_dict = {w.lower(): float(wt) for w, wt in word_weights}
        used_indices = set()
        tokens = text_lc.split()
        merged = []
        i = 0
        while i < len(tokens):
            matched = False
            for phrase, parts in phrase_map.items():
                if i + len(parts) <= len(tokens) and tokens[i:i+len(parts)] == parts:
                    # If all words in the phrase are present in word_weights, merge their weights
                    weights = [word_weight_dict.get(p, 0.0) for p in parts]
                    # Only merge if at least one part is in word_weights
                    if any(abs(w) > 0 for w in weights):
                        merged.append((phrase, sum(weights)))
                        used_indices.update(range(i, i+len(parts)))
                        matched = True
                        break
            if matched:
                i += len(parts)
            else:
                if i not in used_indices:
                    w = tokens[i]
                    if w in word_weight_dict:
                        merged.append((w, word_weight_dict[w]))
                i += 1
        # Remove duplicate single words that are part of merged phrases
        phrase_words = set()
        for phrase, _ in merged:
            for p, parts in phrase_map.items():
                if phrase == p:
                    phrase_words.update(parts)
        final = []
        for feat, wt in merged:
            if feat in phrase_map:
                final.append((feat, wt))
            elif feat not in phrase_words:
                final.append((feat, wt))
        return final

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        num_features: int = LIME_CONFIG["num_features"],
        num_samples: int = LIME_CONFIG["num_samples"]
    ):
        self.class_names = class_names or list(LABEL_MAP.values())
        self.num_features = num_features
        self.num_samples = num_samples
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=r'\W+',
            bow=True
        )
        logger.info(f"LIME Explainer initialized with {num_features} features")

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> Dict:
        num_features = num_features or self.num_features
        num_samples = num_samples or self.num_samples
        explanation = self.explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=2
        )
        probabilities = predict_fn([text])[0]
        predicted_class = int(np.argmax(probabilities))
        word_weights = explanation.as_list(label=predicted_class)
        merged_features = self._merge_phrase_features(text, word_weights)
        positive_features = [
            (word, weight) for word, weight in merged_features if weight > 0
        ]
        negative_features = [
            (word, weight) for word, weight in merged_features if weight < 0
        ]
        positive_features.sort(key=lambda x: x[1], reverse=True)
        negative_features.sort(key=lambda x: abs(x[1]), reverse=True)
        result = {
            'text': text,
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': float(probabilities[predicted_class]),
            'probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            },
            'feature_contributions': [
                {'word': word, 'weight': float(weight)}
                for word, weight in merged_features
            ],
            'positive_features': [
                {'word': word, 'weight': float(weight)}
                for word, weight in positive_features
            ],
            'negative_features': [
                {'word': word, 'weight': float(weight)}
                for word, weight in negative_features
            ],
            'interpretation': self._create_interpretation(
                predicted_class,
                positive_features,
                negative_features
            )
        }
        return result
    
    def _create_interpretation(
        self,
        predicted_class: int,
        positive_features: List[tuple],
        negative_features: List[tuple]
    ) -> str:
        """
        Create a human-readable interpretation of the prediction.
        
        Args:
            predicted_class: The predicted class (0 or 1)
            positive_features: Words supporting the prediction
            negative_features: Words against the prediction
        
        Returns:
            A formatted string explaining the prediction
        """
        # Get the label for the predicted class
        label = self.class_names[predicted_class]
        
        # Build interpretation based on prediction
        if predicted_class == 1:  # Fraudulent
            # Start with the classification
            interpretation = f"This job posting is classified as **{label}**."
            
            # Add suspicious indicators if found
            if positive_features:
                top_words = [f"'{word}'" for word, _ in positive_features[:5]]
                words_str = ', '.join(top_words)
                interpretation += f"\n\n**Suspicious indicators:** {words_str}"
            
            # Add recommendation
            interpretation += "\n\n**Recommendation:** Exercise caution. "
            interpretation += "Verify the company through official channels."
            
        else:  # Legitimate
            # Start with the classification
            interpretation = f"This job posting appears to be **{label}**."
            
            # Add legitimate indicators if found
            if positive_features:
                top_words = [f"'{word}'" for word, _ in positive_features[:5]]
                words_str = ', '.join(top_words)
                interpretation += f"\n\n**Legitimate indicators:** {words_str}"
            
            # Add recommendation
            interpretation += "\n\n**Recommendation:** While legitimate, "
            interpretation += "always verify the company before sharing personal info."
        
        return interpretation
    
    def highlight_text(self, text: str, features: List[Dict]) -> str:
        """
        Create HTML with highlighted important words.
        
        Words that push toward fraudulent are highlighted in red.
        Words that push toward legitimate are highlighted in green.
        
        Args:
            text: The original text
            features: List of feature dictionaries with 'word' and 'weight'
        
        Returns:
            HTML string with color-coded highlights
        """
        # Create a lookup dictionary for word weights
        word_weights = {
            feature['word'].lower(): feature['weight']
            for feature in features
        }
        
        # Split text into words and whitespace
        tokens = re.findall(r'\S+|\s+', text)
        
        # Build result with highlighted words
        result_parts = []
        for token in tokens:
            # Clean the word for lookup
            clean_word = token.lower().strip('.,!?;:"\'()[]{}')
            
            if clean_word in word_weights:
                weight = word_weights[clean_word]
                
                # Red for positive weight (suspicious), green for negative (legitimate)
                if weight > 0:
                    # Red highlight - intensity based on weight
                    opacity = min(abs(weight) * 2, 1)
                    color = f"rgba(255, 99, 71, {opacity})"
                else:
                    # Green highlight - intensity based on weight
                    opacity = min(abs(weight) * 2, 1)
                    color = f"rgba(60, 179, 113, {opacity})"
                
                # Wrap in highlighted span
                highlighted = (
                    f'<span style="background-color: {color}; '
                    f'padding: 2px 4px; border-radius: 3px;">{token}</span>'
                )
                result_parts.append(highlighted)
            else:
                # Not an important word, keep as-is
                result_parts.append(token)
        
        return ''.join(result_parts)
    
    def get_chart_data(self, features: List[Dict], top_n: int = 10) -> Dict:
        """
        Prepare data for creating a bar chart of feature importance.
        
        Args:
            features: List of feature dictionaries
            top_n: Number of top features to include
        
        Returns:
            Dictionary with:
                - words: List of word labels
                - weights: List of weight values
                - colors: List of colors (red for suspicious, teal for legitimate)
        """
        # Sort by absolute weight (importance)
        sorted_features = sorted(
            features,
            key=lambda x: abs(x['weight']),
            reverse=True
        )
        
        # Take top N and reverse for horizontal bar chart (top at top)
        top_features = sorted_features[:top_n][::-1]
        
        # Extract data for chart
        words = [f['word'] for f in top_features]
        weights = [f['weight'] for f in top_features]
        colors = [
            '#FF6B6B' if f['weight'] > 0 else '#4ECDC4'  # Red vs Teal
            for f in top_features
        ]
        
        return {
            'words': words,
            'weights': weights,
            'colors': colors
        }


# =============================================================================
# MAIN - For Testing
# =============================================================================

if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create explainer instance
    explainer = LIMEExplainer()
    
    print("LIME Explainer initialized and ready!")
    print(f"  - Number of features: {explainer.num_features}")
    print(f"  - Number of samples: {explainer.num_samples}")
    print(f"  - Class names: {explainer.class_names}")
