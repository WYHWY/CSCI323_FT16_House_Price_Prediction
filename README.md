# CSCI323_House_Price_Prediction
CSCI323 Group Project - House Price Prediction
# CSCI323 Modern AI - House Price Prediction

## Project Overview
This project aims to predict house prices using the pre-processed Ames Housing dataset. We systematically evaluated three different regression models to compare their predictive performance and robustness.

## Model Performance Comparison (Test Set)
All models were trained on the log-transformed target variable (`log1p`) and evaluated in both the log and dollar spaces.

| Model | MAE (Dollar) | RMSE (Dollar) | R² (Dollar) | R² (Log) |
| :--- | :--- | :--- | :--- | :--- |
| Ridge Regression | $16,563.33 | $72,986.39 | 0.2306 | 0.8537 |
| Decision Tree | $23,623.80 | $38,299.20 | 0.7881 | 0.7883 |
| **Random Forest** | **$15,720.09** | **$25,901.91** | **0.9031** | **0.8954** |

*Note: Feature scaling (`StandardScaler`) was only applied to the Ridge model, as tree-based models are invariant to feature scales.*

## Key Insights & Feature Importance
*   **Random Forest** achieved the best overall performance, with the lowest MAE and highest R², demonstrating the effectiveness of ensemble learning in reducing variance compared to a single Decision Tree.
*   **Feature Importance**: Based on Mean Decrease in Impurity (MDI), `Overall Qual` (overall material and finish quality) is the most critical feature, accounting for approximately **58.3%** of the Random Forest's relative importance, followed by `Gr Liv Area` (above grade living area). 
*   *Disclaimer: Feature importance scores represent the model's internal reliance and associations, not direct causal relationships.*

## Repository Structure
*   `code/`
    *   `ridge_model.py`: Implementation and evaluation of the Ridge Regression baseline.
    *   `tree_models.py`: Implementation and evaluation of Decision Tree and Random Forest models.
*   `data/`
    *   `train_encoded.csv`: Pre-processed training dataset.
    *   `test_encoded.csv`: Pre-processed testing dataset.
*   `docs/`
    *   Contains data summaries, feature names, and assignment splits from the data processing team.
