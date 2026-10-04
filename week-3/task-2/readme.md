# Week 3 Task 2: NLP — Text/Sentiment Classification

**Author:** Muhammad Ahmad
**Program:** Evolvix AI/ML Internship

## Project Overview
This repository contains the implementation and comparison of two distinct approaches for sentiment analysis using the IMDb movie reviews dataset. The goal of this task is to evaluate a classical machine learning model against a fine-tuned Transformer model. 

## Models Compared
1. **Classical Machine Learning Approach:** 
   - **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
   - **Classifier:** Logistic Regression
2. **Transformer Approach (Stretch Goal):**
   - **Model:** Fine-tuned Longformer (`ahmed792002/Finetuning_Longformer_IMDb_movie_reviews_Classification`)
   - Loaded and evaluated using the Hugging Face `transformers` pipeline.

## Dataset
- **Source:** IMDb Movie Reviews 
- **Task:** Binary Text Classification (Positive vs. Negative sentiment)

## Evaluation & Results
Both models were evaluated on a balanced subset of the test data. The performance metrics are as follows:

| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | 0.71 (71%) | 0.72 |
| **Hugging Face Transformer** | **0.94 (94%)** | **0.94** |

## Conclusion
The fine-tuned Hugging Face Transformer significantly outperformed the classical TF-IDF model in both accuracy and F1-score. While TF-IDF relies on basic keyword frequency and struggles with complex sentence structures, the Transformer model effectively captures the contextual meaning and deeper nuances of the movie reviews, leading to highly accurate sentiment predictions.
