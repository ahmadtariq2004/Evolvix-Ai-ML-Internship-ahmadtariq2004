# Week 1 Task 2 - First Machine Learning Model

**Author:** Muhammad Ahmad
**Dataset:** Automobile Fuel Economy (`cars.csv`)

## Project Overview
This project is part of the EvolviX AI/ML Internship. The objective of this task is to train, evaluate, and save a baseline machine learning regression model to predict fuel economy (MPG).

## Steps Performed
1. **Data Preprocessing:** Handled missing values by replacing them with column means and applied one-hot encoding to the categorical `brand` column.
2. **Data Splitting:** Split the dataset into training (80%) and testing (20%) sets to ensure fair evaluation and prevent data leakage.
3. **Model Training:** Trained a baseline **Linear Regression** model using `scikit-learn` to predict the continuous `mpg` target variable.
4. **Model Evaluation:** Evaluated the model's performance on the test set using Root Mean Squared Error (RMSE) and R-squared (R2).
5. **Model Saving:** Exported the trained model as a `.joblib` file (`linear_regression_model.joblib`).

## Model Performance & Results
| Metric | Value | Explanation |
|---|---|---|
| **RMSE** | 3.31 | The average error between the predicted and actual MPG values is 3.31 MPG. A lower RMSE indicates a better fit. |
| **R-squared (R2)** | 0.82 | The model explains 82% of the variance in the fuel economy (MPG). This indicates strong predictive performance for a baseline model. |

## Files Included
- `ml_model.ipynb`: Jupyter Notebook with the code for data preprocessing, model training, and evaluation.
- `linear_regression_model.joblib`: The saved Linear Regression model ready for deployment.
