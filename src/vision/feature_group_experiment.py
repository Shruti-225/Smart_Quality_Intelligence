
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

import vision_model as vm
import normalization_experiment as ne


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {
    "All 103 features": list(range(103)),

    "Structural only": (
        [37, 38] + list(range(39, 103))
    ),

    "Histogram + Spatial": (
        list(range(5, 37)) + list(range(39, 103))
    ),

    "Without intensity statistics": list(range(5, 103)),
}


# ============================================================
# FEATURE MATRIX
# ============================================================

def extract_group_features(paths, description):

    features = ne.build_normalized_matrix(
        paths,
        "Percentile",
        description,
    )

    return features


# ============================================================
# MODEL EXPERIMENT
# ============================================================

def evaluate_group(
    group_name,
    feature_indices,
    fit_features,
    calibration_features,
    test_features,
    test_labels,
):

    print("\n" + "=" * 60)
    print(f"FEATURE GROUP: {group_name}")
    print("=" * 60)

    X_train = fit_features[:, feature_indices]
    X_calibration = calibration_features[:, feature_indices]
    X_test = test_features[:, feature_indices]

    print("Selected features:", len(feature_indices))

    model = vm.train_anomaly_detector(X_train)

    calibration_scores = vm.calculate_scores(
        model,
        X_calibration,
    )

    threshold = vm.determine_threshold(
        calibration_scores
    )

    test_scores = vm.calculate_scores(
        model,
        X_test,
    )

    predictions = (
        test_scores >= threshold
    ).astype(int)

    auc = roc_auc_score(
        test_labels,
        test_scores,
    )

    precision = precision_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    balanced_acc = balanced_accuracy_score(
        test_labels,
        predictions,
    )

    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=[0, 1],
    )

    print(f"Threshold: {threshold:.6f}")
    print(f"ROC-AUC: {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"Balanced Accuracy: {balanced_acc:.6f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "Feature Group": group_name,
        "Feature Count": len(feature_indices),
        "Threshold": threshold,
        "ROC_AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Balanced Accuracy": balanced_acc,
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========== FEATURE GROUP EXPERIMENT ==========")

    training_paths = vm.collect_training_images()

    testing_paths, test_labels, categories = (
        vm.collect_testing_images()
    )

    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE,
    )

    print("\nTraining fit:", len(fit_paths))
    print("Calibration:", len(calibration_paths))
    print("Testing:", len(testing_paths))

    # Extract each dataset once.
    fit_features = extract_group_features(
        fit_paths,
        "training images",
    )

    calibration_features = extract_group_features(
        calibration_paths,
        "calibration images",
    )

    test_features = extract_group_features(
        testing_paths,
        "testing images",
    )

    results = []

    for group_name, indices in FEATURE_GROUPS.items():

        result = evaluate_group(
            group_name,
            indices,
            fit_features,
            calibration_features,
            test_features,
            test_labels,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n========== FEATURE GROUP COMPARISON ==========")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n========== EXPERIMENT COMPLETE ==========")
    print("No existing model artifacts were modified.")


if __name__ == "__main__":
    main()
