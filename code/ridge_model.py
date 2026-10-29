import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


# 3. Build the model pipeline
print("Training Ridge Regression model...")
model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)


# 4. Train using the log-transformed target
model.fit(X_train, np.log1p(y_train))


# 5. Predict and convert predictions back to dollars
pred_log = model.predict(X_test)
pred = np.expm1(pred_log)


# 6. Calculate evaluation metrics
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2_dollar = r2_score(y_test, pred)
r2_log = r2_score(np.log1p(y_test), pred_log)


# 7. Print results
print("\n" + "=" * 40)
print("Ridge Regression Results (Test Set):")
print("=" * 40)
print(f"MAE (Dollar):  ${mae:,.2f}")
print(f"RMSE (Dollar): ${rmse:,.2f}")
print(f"R² (Dollar):   {r2_dollar:.4f}")
print(f"R² (Log):      {r2_log:.4f}")
print("=" * 40)

print("\n[Reference Check]")
print("Handoff README Ridge MAE: $16,563")
print("Handoff README Ridge R² (Log): 0.8537")


# 8. Save per-house predictions for error analysis
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

predictions = pd.DataFrame({
    "row_id": test["row_id"],
    "actual_sale_price": y_test,
    "ridge_predicted_price": pred,
    "ridge_abs_error": np.abs(y_test - pred)
})

output_file = results_dir / "ridge_test_predictions.csv"
predictions.to_csv(output_file, index=False)

print(f"\nPer-house predictions saved to: {output_file}")
