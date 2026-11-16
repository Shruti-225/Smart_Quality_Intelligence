
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

import vision_model as vm
import normalization_experiment as ne


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

NORMALIZATION_METHOD = "Percentile"
THRESHOLD_PERCENTILE = 80

# Exclude the first five intensity statistics.
FEATURE_INDICES = list(range(5, 103))


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n========== DEFECT CATEGORY ANALYSIS ==========")

    # Collect dataset
    training_paths = vm.collect_training_images()

    testing_paths, test_labels, categories = (
        vm.collect_testing_images()
    )

    test_labels = np.asarray(test_labels)
    categories = np.asarray(categories)

    # Same training/calibration split as previous experiments
    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE,
    )

    print("\nTraining fit:", len(fit_paths))
    print("Calibration:", len(calibration_paths))
    print("Testing:", len(testing_paths))

    # Extract normalized features
    fit_features = ne.build_normalized_matrix(
        fit_paths,
        NORMALIZATION_METHOD,
        "training images",
    )

    calibration_features = ne.build_normalized_matrix(
        calibration_paths,
        NORMALIZATION_METHOD,
        "calibration images",
    )

    test_features = ne.build_normalized_matrix(
        testing_paths,
        NORMALIZATION_METHOD,
        "testing images",
    )

    # Select features
    X_fit = fit_features[:, FEATURE_INDICES]
    X_calibration = calibration_features[:, FEATURE_INDICES]
    X_test = test_features[:, FEATURE_INDICES]

    print("\nSelected features:", len(FEATURE_INDICES))

    # Train temporary model
    model = vm.train_anomaly_detector(X_fit)

    # Calculate calibration scores
    calibration_scores = vm.calculate_scores(
        model,
        X_calibration,
    )

    threshold = np.percentile(
        calibration_scores,
        THRESHOLD_PERCENTILE,
    )

    print(f"\nThreshold percentile: {THRESHOLD_PERCENTILE}")
    print(f"Calculated threshold: {threshold:.6f}")

    # Predict test images
    test_scores = vm.calculate_scores(
        model,
        X_test,
    )

    predictions = (
        test_scores >= threshold
    ).astype(int)

    # Overall metrics
    print("\n========== OVERALL PERFORMANCE ==========")

    print(
        "ROC-AUC:",
        round(roc_auc_score(test_labels, test_scores), 6),
    )

    print(
        "Precision:",
        round(precision_score(
            test_labels, predictions, zero_division=0
        ), 4),
    )

    print(
        "Recall:",
        round(recall_score(
            test_labels, predictions, zero_division=0
        ), 4),
    )

    print(
        "F1:",
        round(f1_score(
            test_labels, predictions, zero_division=0
        ), 4),
    )

    print("\nOverall confusion matrix:")
    print(confusion_matrix(
        test_labels,
        predictions,
        labels=[0, 1],
    ))

    # Category-wise analysis
    print("\n========== DEFECT CATEGORY RESULTS ==========")

    results = []

    for category in sorted(set(categories)):

        mask = (
            (categories == category) &
            (test_labels == 1)
        )

        actual = test_labels[mask]
        predicted = predictions[mask]

        if len(actual) == 0:
            continue

        detected = int(np.sum(predicted == 1))
        missed = int(np.sum(predicted == 0))

        category_recall = recall_score(
            actual,
            predicted,
            zero_division=0,
        )

        category_precision = precision_score(
            actual,
            predicted,
            zero_division=0,
        )

        category_f1 = f1_score(
            actual,
            predicted,
            zero_division=0,
        )

        print("\n-----------------------------------")
        print("Defect category:", category)
        print("Total images:", len(actual))
        print("Detected:", detected)
        print("Missed:", missed)
        print(f"Recall: {category_recall:.4f}")
        print(f"Precision: {category_precision:.4f}")
        print(f"F1-score: {category_f1:.4f}")

        results.append({
            "Defect Category": category,
            "Total Images": len(actual),
            "Detected": detected,
            "Missed": missed,
            "Recall": category_recall,
            "Precision": category_precision,
            "F1": category_f1,
        })

    # Final comparison table
    results_df = pd.DataFrame(results)

    print("\n========== CATEGORY COMPARISON TABLE ==========")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n========== EXPERIMENT COMPLETE ==========")
    print("No production model files were modified.")


if __name__ == "__main__":
    main()
