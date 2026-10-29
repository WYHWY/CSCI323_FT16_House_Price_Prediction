import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
print("Loading data...")
# Note: Using relative path for GitHub repository structure
train = pd.read_csv("../data/train_encoded.csv")
test = pd.read_csv("../data/test_encoded.csv")

# 2. Split features and target variable
NON_FEATURES = ["row_id", "SalePrice", "fold"]
FEATURES = [c for c in train.columns if c not in NON_FEATURES]

X_train, y_train = train[FEATURES], train["SalePrice"]
X_test, y_test = test[FEATURES], test["SalePrice"]

# 3. Model pipeline (StandardScaler + Ridge)
print("Training Ridge Regression model...")
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

# 4. Train model (Apply log1p transformation to target variable)
model.fit(X_train, np.log1p(y_train))

# 5. Predict and convert back to dollar space
pred_log = model.predict(X_test)     # Log space
pred = np.expm1(pred_log)            # Dollar space

# 6. Calculate evaluation metrics
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2_dollar = r2_score(y_test, pred)
r2_log = r2_score(np.log1p(y_test), pred_log)

# 7. Print results
print("\n" + "="*40)
print("Ridge Regression Results (Test Set):")
print("="*40)
print(f"MAE (Dollar):  ${mae:,.2f}")
print(f"RMSE (Dollar): ${rmse:,.2f}")
print(f"R² (Dollar):   {r2_dollar:.4f}")
print(f"R² (Log):      {r2_log:.4f}")
print("="*40)

# 8. Extract and display Feature Importance (for report analysis)
print("\nTop 5 Most Important Features (Ridge):")
# For Ridge, we use the absolute value of coefficients to represent importance
importances = np.abs(model.named_steps['ridge'].coef_)
indices = np.argsort(importances)[::-1]
for i in range(5):
    clean_name = FEATURES[indices[i]].split("__")[-1] # Strip prefix
    importance_score = importances[indices[i]]
    print(f"{i+1}. {clean_name}: {importance_score:.4f}")
