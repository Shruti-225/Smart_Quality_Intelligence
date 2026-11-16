
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "reports"
    / "vision"
    / "local_texture_feature_values.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "reports" / "vision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 300
CALIBRATION_PERCENTILE = 95


# ============================================================
# FEATURE CONFIGURATIONS
# ============================================================

FEATURE_SETS = {
    "All 10 Features": [
        "lbp_entropy",
        "lbp_uniformity",
        "gradient_mean",
        "gradient_std",
        "gradient_p90",
        "local_contrast_mean",
        "local_contrast_std",
        "local_contrast_p90",
        "edge_density",
        "edge_count",
    ],

    "Selected 6 Features": [
        "local_contrast_std",
        "gradient_p90",
        "lbp_uniformity",
        "edge_density",
        "local_contrast_p90",
        "gradient_mean",
    ],

    "Local Contrast Only": [
        "local_contrast_mean",
        "local_contrast_std",
        "local_contrast_p90",
    ],

    "Gradient Only": [
        "gradient_mean",
        "gradient_std",
        "gradient_p90",
    ],

    "LBP and Edge Only": [
        "lbp_entropy",
        "lbp_uniformity",
        "edge_density",
    ],
}


# ============================================================
# LOAD DATA
# ============================================================

print("\n========== COMBINED TEXTURE FEATURE EXPERIMENT ==========\n")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = ["Group", "Image"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

all_features = sorted({
    feature
    for feature_list in FEATURE_SETS.values()
    for feature in feature_list
})

missing = [
    feature for feature in all_features
    if feature not in df.columns
]

if missing:
    raise ValueError(f"Missing features: {missing}")

df["Group"] = df["Group"].astype(str).str.strip()

df[all_features] = df[all_features].apply(
    pd.to_numeric,
    errors="coerce"
)

df = df.dropna(subset=all_features).copy()


# ============================================================
# IDENTIFY DATA SPLITS
# ============================================================

training = df[df["Group"] == "Training Fit Good"].copy()
calibration = df[df["Group"] == "Calibration Good"].copy()

test = df[
    df["Group"].isin(
        [
            "Test Good",
            "manipulated_front",
            "scratch_head",
            "scratch_neck",
            "thread_side",
            "thread_top",
        ]
    )
].copy()

if len(training) != 256:
    raise ValueError(
        f"Expected 256 training images, found {len(training)}"
    )

if len(calibration) != 64:
    raise ValueError(
        f"Expected 64 calibration images, found {len(calibration)}"
    )

if len(test) != 160:
    raise ValueError(
        f"Expected 160 test images, found {len(test)}"
    )

y_test = (
    test["Group"] != "Test Good"
).astype(int).to_numpy()

print("Training Fit Good:", len(training))
print("Calibration Good:", len(calibration))
print("Test images:", len(test))
print("Test Good:", int((y_test == 0).sum()))
print("Test Defective:", int((y_test == 1).sum()))


# ============================================================
# MODEL EVALUATION
# ============================================================

results = []
category_results = []

for set_name, features in FEATURE_SETS.items():

    print("\n" + "=" * 65)
    print("FEATURE SET:", set_name)
    print("Number of features:", len(features))
    print("Features:", features)
    print("=" * 65)

    X_train = training[features].to_numpy(dtype=float)
    X_calibration = calibration[features].to_numpy(dtype=float)
    X_test = test[features].to_numpy(dtype=float)

    # Temporary model. No production artifact is loaded or saved.
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        contamination="auto",
    )

    model.fit(X_train)

    # Higher score = more anomalous.
    calibration_scores = -model.decision_function(
        X_calibration
    )

    test_scores = -model.decision_function(X_test)

    # Threshold determined only from normal calibration data.
    threshold = np.percentile(
        calibration_scores,
        CALIBRATION_PERCENTILE
    )

    predictions = (
        test_scores >= threshold
    ).astype(int)

    auc = roc_auc_score(y_test, test_scores)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    print(f"\nCalibration threshold: {threshold:.6f}")
    print(f"ROC-AUC:              {auc:.6f}")
    print(f"Precision:            {precision:.6f}")
    print(f"Recall:               {recall:.6f}")
    print(f"F1-score:             {f1:.6f}")
    print(f"Balanced accuracy:    {balanced_acc:.6f}")

    print("\nConfusion Matrix:")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    results.append({
        "feature_set": set_name,
        "feature_count": len(features),
        "features": ", ".join(features),
        "threshold": threshold,
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "balanced_accuracy": balanced_acc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })

    # Category-wise detection for diagnostic purposes.
    for category in [
        "manipulated_front",
        "scratch_head",
        "scratch_neck",
        "thread_side",
        "thread_top",
    ]:

        category_mask = (
            test["Group"].to_numpy() == category
        )

        category_predictions = predictions[category_mask]

        detected = int(category_predictions.sum())
        total = int(category_mask.sum())

        category_recall = (
            detected / total if total else 0
        )

        print(
            f"{category}: "
            f"{detected}/{total} detected "
            f"({category_recall:.2%})"
        )

        category_results.append({
            "feature_set": set_name,
            "category": category,
            "detected": detected,
            "total": total,
            "recall": category_recall,
        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

category_df = pd.DataFrame(category_results)

results_path = (
    OUTPUT_DIR / "combined_texture_model_comparison.csv"
)

category_path = (
    OUTPUT_DIR / "combined_texture_category_comparison.csv"
)

results_df.to_csv(results_path, index=False)
category_df.to_csv(category_path, index=False)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n========== FINAL MODEL COMPARISON ==========\n")

display_columns = [
    "feature_set",
    "feature_count",
    "roc_auc",
    "precision",
    "recall",
    "f1_score",
    "balanced_accuracy",
]

print(
    results_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n========== REPORTS SAVED ==========\n")

print(results_path)
print(category_path)

print("\nEXPERIMENT COMPLETE")
print(
    "No production model, dashboard, "
    "or saved model artifacts were changed."
)
