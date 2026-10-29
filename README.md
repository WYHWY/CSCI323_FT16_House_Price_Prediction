# CSCI323 Modern AI - House Price Prediction

## Project Overview

This project predicts house prices using the pre-processed Ames Housing dataset. We compare the predictive performance of three regression models: Ridge Regression, Decision Tree, and Random Forest.

## Setup and Execution

Make sure Python 3 is installed. From the repository root, install the required packages:

```bash
pip install pandas numpy scikit-learn
```

Run the models from the repository root:

```bash
python code/ridge_model.py
python code/tree_models.py
```

The scripts use `pathlib` to locate the data files relative to their own locations.

## Model Performance Comparison

All models were trained using the log-transformed target, `log1p(SalePrice)`. Results are reported in both dollar and log spaces.

| Model | MAE (Dollar) | RMSE (Dollar) | R² (Dollar) | R² (Log) |
| :--- | ---: | ---: | ---: | ---: |
| Ridge Regression | $16,563.33 | $72,986.39 | 0.2306 | 0.8537 |
| Decision Tree | $23,623.80 | $38,299.20 | 0.7881 | 0.7883 |
| **Random Forest** | **$15,720.09** | **$25,901.91** | **0.9031** | **0.8954** |

*Note: `StandardScaler` was applied only to Ridge Regression. Tree-based models do not require feature scaling.*

## Key Findings and Feature Importance

- Random Forest achieved the lowest MAE and the highest R² among the three models on the test set.
- Based on Mean Decrease in Impurity (MDI), `Overall Qual` was the most important feature in the Random Forest model, accounting for approximately 58.3% of its total feature importance.
- Feature importance describes how the fitted model uses features. It does not establish that a feature causes house prices to change.

## Repository Structure

- `code/`
  - `ridge_model.py`: trains and evaluates the Ridge Regression model.
  - `tree_models.py`: trains and evaluates the Decision Tree and Random Forest models.
- `data/`
  - `train_encoded.csv`: pre-processed training dataset.
  - `test_encoded.csv`: pre-processed test dataset.
- `docs/`
  - Data summaries, feature names, and split information from the data processing team.
