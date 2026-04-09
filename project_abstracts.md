# Final Semester Project - Abstract Variations
## Topic: Online Recruitment Fraud Detection

Based on the research paper: "Online Recruitment Fraud (ORF) Detection Using Deep Learning Approaches" by Natasha Akram et al.

---

## Abstract 1: Simplified BERT-based Fake Job Detection with Explainable AI

**Title:** Explainable Fake Job Posting Detection Using BERT and LIME

**Abstract:**
The proliferation of online job portals has led to an increase in fraudulent job postings that deceive job seekers and compromise their personal information. This project presents an accessible and interpretable approach to detecting fake job postings using the BERT (Bidirectional Encoder Representations from Transformers) model. We utilize the publicly available Employment Scam Aegean Dataset (EMSCAD) and apply SMOTE (Synthetic Minority Oversampling Technique) to address the inherent class imbalance between legitimate and fraudulent postings. The key contribution of this work is the integration of LIME (Local Interpretable Model-agnostic Explanations) to provide human-understandable explanations for each prediction, making the model's decisions transparent. Our approach focuses on text-based features including job titles, descriptions, and company profiles. The model is evaluated using balanced accuracy, precision, recall, and F1-score metrics. This project demonstrates that effective fraud detection can be achieved with explainable outcomes, enabling recruiters and job seekers to understand why a posting is flagged as suspicious.

**Key Features:**
- Single transformer model (BERT) for simplicity
- SMOTE for handling imbalanced data
- LIME for model explainability
- Focus on text features only
- Web-based demo interface

---

## Abstract 2: Comparative Analysis of Machine Learning Models for Job Fraud Detection

**Title:** A Comparative Study of Traditional ML and Deep Learning for Online Job Fraud Detection

**Abstract:**
Online recruitment fraud has become a significant concern as cybercriminals exploit job portals to conduct scams. This project conducts a comprehensive comparative analysis between traditional machine learning algorithms and deep learning approaches for detecting fraudulent job postings. We implement and compare five models: Logistic Regression, Random Forest, Support Vector Machine (SVM), LSTM (Long Short-Term Memory), and a fine-tuned DistilBERT model. The study utilizes the EMSCAD dataset with preprocessing techniques including text cleaning, TF-IDF vectorization for traditional models, and tokenization for deep learning models. To handle the severe class imbalance (approximately 95% real vs 5% fake jobs), we implement Random Oversampling and SMOTE techniques. Our evaluation focuses on practical metrics including balanced accuracy, recall (to minimize missed fraudulent posts), and inference time (for real-world applicability). The results provide insights into the trade-offs between model complexity, accuracy, and computational requirements, offering practical recommendations for deploying fraud detection systems in resource-constrained environments.

**Key Features:**
- Side-by-side comparison of 5 different models
- Traditional ML vs Deep Learning analysis
- Multiple oversampling techniques compared
- Focus on practical deployment considerations
- Clear visualization of results

---

## Abstract 3: Feature Engineering and Ensemble Approach for Fake Job Detection

**Title:** Detecting Fake Job Postings Using Feature Engineering and Ensemble Classification

**Abstract:**
The increasing sophistication of online job scams necessitates robust detection mechanisms to protect job seekers. This project proposes a feature-engineered ensemble approach for identifying fraudulent job postings that balances effectiveness with interpretability. We extract and analyze multiple feature categories: (1) textual features using TF-IDF from job descriptions and requirements, (2) metadata features such as presence of company logo, salary information, and location details, and (3) linguistic features including text length, punctuation patterns, and readability scores. These engineered features are fed into an ensemble classifier combining Random Forest, Gradient Boosting, and Logistic Regression using a soft voting mechanism. The EMSCAD dataset is preprocessed and balanced using SMOTE-ENN (SMOTE combined with Edited Nearest Neighbors) to handle class imbalance while removing noisy samples. Feature importance analysis reveals which job posting characteristics are most indicative of fraud, providing actionable insights for manual verification. Our approach achieves competitive accuracy while maintaining full interpretability, making it suitable for practical deployment where understanding model decisions is crucial.

**Key Features:**
- Comprehensive feature engineering
- Ensemble of interpretable models
- SMOTE-ENN for better data quality
- Feature importance analysis
- No black-box deep learning (fully explainable)

---

## Comparison Summary

| Aspect | Abstract 1 | Abstract 2 | Abstract 3 |
|--------|------------|------------|------------|
| **Complexity** | Medium | Medium-High | Low-Medium |
| **Model Type** | Deep Learning (BERT) | Multiple (ML + DL) | Traditional ML Ensemble |
| **Explainability** | LIME-based | Comparative metrics | Feature importance |
| **Implementation Difficulty** | Moderate | Higher (multiple models) | Easier |
| **Computational Resources** | GPU recommended | GPU required | CPU sufficient |
| **Best For** | NLP-focused projects | Research-oriented | Practical applications |

---

## Recommended Datasets

1. **EMSCAD (Employment Scam Aegean Dataset)** - Primary dataset, publicly available
2. **Kaggle Fake Job Postings Dataset** - Alternative/supplementary dataset

## Suggested Tools & Libraries

- Python, Scikit-learn, Pandas, NumPy
- Hugging Face Transformers (for BERT/DistilBERT)
- Imbalanced-learn (for SMOTE variants)
- LIME/SHAP (for explainability)
- Streamlit/Flask (for demo interface)
