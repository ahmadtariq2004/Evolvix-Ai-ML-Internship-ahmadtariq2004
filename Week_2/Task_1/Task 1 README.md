# Week 2 - Task 1: Classification Model Comparison

This folder contains the implementation for Week 2, Task 1 of the Evolvix AI/ML Internship. The objective of this task was to build, train, and compare different classification models using the Breast Cancer dataset.

## Models Evaluated
The following machine learning models were trained and evaluated on the dataset:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

## Performance Metrics
The models were evaluated based on Precision, Recall, F1-Score, and ROC-AUC. Below is the performance comparison on the test set:

| Model | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9636 | 0.9815 | 0.9725 | **0.9978** |
| **Decision Tree** | **0.9712** | 0.9352 | 0.9528 | 0.9438 |
| **Random Forest** | 0.9640 | **0.9907** | **0.9772** | 0.9968 |

## Conclusion
Based on the ROC-AUC score, the **Logistic Regression** model proved to be the best-performing model with the highest AUC of **0.9978**. 

However, the **Random Forest Classifier** also demonstrated excellent performance, achieving the highest Recall (0.9907) and F1-Score (0.9772). While Logistic Regression provides a strong baseline and high interpretability, Random Forest remains a highly robust candidate for deployment due to its ensemble nature and superior discriminative power.
