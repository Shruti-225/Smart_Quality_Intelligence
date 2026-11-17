
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "production_cleaned.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "random_forest_model.pkl"
FEATURES_PATH = BASE_DIR / "outputs" / "models" / "selected_features.pkl"

# Load original data
df = pd.read_csv(DATA_PATH)

# Load trained model and selected features
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

X = df[features]
y_true = df["quality_label"].astype(int)

# Generate predictions
y_pred = model.predict(X)

# Get probability of defective class
probabilities = model.predict_proba(X)[:, 1]

# Apply the dashboard's 0.40 threshold
y_pred_threshold = (probabilities >= 0.40).astype(int)

print("\n========== AI MODEL EVALUATION ==========")
print("Total records:", len(df))
print("Selected features:", len(features))

print("\nActual class distribution:")
print(y_true.value_counts().sort_index())

print("\nPredicted class distribution:")
print(pd.Series(y_pred_threshold).value_counts().sort_index())

print("\n========== PERFORMANCE METRICS ==========")
print("Accuracy:", round(accuracy_score(y_true, y_pred_threshold), 4))
print("Precision:", round(precision_score(y_true, y_pred_threshold, zero_division=0), 4))
print("Recall:", round(recall_score(y_true, y_pred_threshold, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_true, y_pred_threshold, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_true, probabilities), 4))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_true, y_pred_threshold))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_true, y_pred_threshold, zero_division=0))

print("\nEvaluation completed.")
