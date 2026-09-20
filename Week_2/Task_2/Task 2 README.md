# Week 2 - Task 2: Feature Engineering & Hyperparameter Tuning

This repository contains the implementation for Week 2, Task 2 of the Evolvix AI/ML Internship. The objective of this task was to improve a baseline machine learning model through feature engineering and hyperparameter tuning.

## Techniques Applied

### 1. Feature Engineering
To improve the model's predictive power, the following new features were engineered from the existing dataset:
* **Power-to-Weight Ratio:** Calculated by dividing Horsepower (`hp`) by Weight (`weightlbs`).
* **Engine Category:** Grouped numerical engine metrics (cylinders/hp) into categorical bins (e.g., Small, Medium, Large, V-Engine).
* **Car Age:** Derived by subtracting the car's manufacturing year from the current year.

### 2. Hyperparameter Tuning
Applied `GridSearchCV` on a `GradientBoostingRegressor` pipeline to find the optimal configuration. The best parameters found were:
* `learning_rate`: 0.1
* `max_depth`: 5
* `min_samples_leaf`: 1
* `min_samples_split`: 4
* `n_estimators`: 100

## Performance Comparison (Before vs. After)

The tuned model was evaluated against the baseline model (from Task 1) to quantify the improvements.

| Metric | Baseline Model | Tuned Model | Improvement |
| :--- | :--- | :--- | :--- |
| **R-squared (R²)** | 0.8732 | 0.8823 | + 0.0091 |
| **Mean Absolute Error (MAE)** | 2.01 | 1.99 | + 0.02 |
| **Root Mean Squared Error (RMSE)** | 2.81 | 2.71 | + 0.10 |

## Conclusion
The tuned model with engineered features successfully outperformed the baseline model across all evaluated metrics. It demonstrated a higher R-squared value, indicating better variance explanation, along with reduced error rates (lower MAE and RMSE).
