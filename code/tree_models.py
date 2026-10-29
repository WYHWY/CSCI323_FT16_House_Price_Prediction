import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Use the folder containing this script
ROOT = Path(__file__).resolve().parent


# 1. Load data
print("Loading data...")
train = pd.read_csv(ROOT / "train_encoded.csv")
test = pd.read_csv(ROOT / "test_encoded.csv")


# 2. Split features and target variable
NON_FEATURES = ["row_id", "SalePrice", "fold"]
FEATURES = [column for column in train.columns if column not in NON_FEATURES]

X_train = train[FEATURES]
y_train = train["SalePrice"]

X_test = test[FEATURES]
y_test = test["SalePrice"]


# 3. Train Random Forest
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, np.log1p(y_train))


# Predict and evaluate Random Forest
rf_pred_log = rf_model.predict(X_test)
rf_pred = np.expm1(rf_pred_log)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2_dollar = r2_score(y_test, rf_pred)
rf_r2_log = r2_score(np.log1p(y_test), rf_pred_log)


# 4. Train Decision Tree
print("Training Decision Tree model...")
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, np.log1p(y_train))


# Predict and evaluate Decision Tree
dt_pred_log = dt_model.predict(X_test)
dt_pred = np.expm1(dt_pred_log)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2_dollar = r2_score(y_test, dt_pred)
dt_r2_log = r2_score(np.log1p(y_test), dt_pred_log)


# 5. Print results
print("\n" + "=" * 50)
print("Tree Models Results (Test Set):")
print("=" * 50)

print(f"Random Forest - MAE (Dollar):  ${rf_mae:,.2f}")
print(f"Random Forest - RMSE (Dollar): ${rf_rmse:,.2f}")
print(f"Random Forest - R² (Dollar):   {rf_r2_dollar:.4f}")
print(f"Random Forest - R² (Log):      {rf_r2_log:.4f}")

print("-" * 50)

print(f"Decision Tree - MAE (Dollar):  ${dt_mae:,.2f}")
print(f"Decision Tree - RMSE (Dollar): ${dt_rmse:,.2f}")
print(f"Decision Tree - R² (Dollar):   {dt_r2_dollar:.4f}")
print(f"Decision Tree - R² (Log):      {dt_r2_log:.4f}")
print("=" * 50)


# 6. Display Random Forest feature importance
print("\nTop 5 Most Important Features (Random Forest):")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(5):
    feature_name = FEATURES[indices[i]]
    importance_score = importances[indices[i]]
    print(f"{i + 1}. {feature_name}: {importance_score:.4f}")


# 7. Save per-house predictions for error analysis
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

predictions = pd.DataFrame({
    "row_id": test["row_id"],
    "actual_sale_price": y_test,
    "rf_predicted_price": rf_pred,
    "rf_abs_error": np.abs(y_test - rf_pred),
    "dt_predicted_price": dt_pred,
    "dt_abs_error": np.abs(y_test - dt_pred)
})

output_file = results_dir / "tree_test_predictions.csv"
predictions.to_csv(output_file, index=False)

print(f"\nPer-house predictions saved to: {output_file}")
