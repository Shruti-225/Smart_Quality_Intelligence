
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
# EXPERIMENT SETTINGS
# ============================================================

THRESHOLD_PERCENTILES = [95, 90, 85, 80]

FEATURE_GROUPS = {
    "All 103 features": list(range(103)),
    "Structural only": [37, 38] + list(range(39, 103)),
    "Histogram + Spatial": list(range(5, 37)) + list(range(39, 103)),
    "Without intensity statistics": list(range(5, 103)),
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(paths, description):

    return ne.build_normalized_matrix(
        paths,
        "Percentile",
        description,
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(
    group_name,
    feature_indices,
    fit_features,
    calibration_features,
    test_features,
    test_labels,
):

    print("\n" + "=" * 65)
    print("FEATURE GROUP:", group_name)
    print("=" * 65)

    X_fit = fit_features[:, feature_indices]
    X_calibration = calibration_features[:, feature_indices]
    X_test = test_features[:, feature_indices]

    print("Feature count:", len(feature_indices))

    model = vm.train_anomaly_detector(X_fit)

    calibration_scores = vm.calculate_scores(
        model,
        X_calibration,
    )

    test_scores = vm.calculate_scores(
        model,
        X_test,
    )

    auc = roc_auc_score(test_labels, test_scores)

    print(f"ROC-AUC: {auc:.6f}")

    results = []

    for percentile in THRESHOLD_PERCENTILES:

        threshold = np.percentile(
            calibration_scores,
            percentile,
        )

        predictions = (
            test_scores >= threshold
        ).astype(int)

        matrix = confusion_matrix(
            test_labels,
            predictions,
            labels=[0, 1],
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

        tn, fp, fn, tp = matrix.ravel()

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        print("\n-----------------------------------")
        print(f"Calibration percentile: {percentile}")
        print(f"Threshold: {threshold:.6f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")

        print("Confusion Matrix:")
        print(matrix)

        results.append({
            "Feature Group": group_name,
            "Percentile": percentile,
            "Threshold": threshold,
            "ROC_AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Balanced Accuracy": balanced_acc,
            "False Positive Rate": false_positive_rate,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========== THRESHOLD SENSITIVITY EXPERIMENT ==========")

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

    fit_features = extract_features(
        fit_paths,
        "training images",
    )

    calibration_features = extract_features(
        calibration_paths,
        "calibration images",
    )

    test_features = extract_features(
        testing_paths,
        "testing images",
    )

    all_results = []

    for group_name, feature_indices in FEATURE_GROUPS.items():

        results = evaluate_thresholds(
            group_name,
            feature_indices,
            fit_features,
            calibration_features,
            test_features,
            test_labels,
        )

        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    print("\n\n========== COMPLETE THRESHOLD COMPARISON ==========")

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
