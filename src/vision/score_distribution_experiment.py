
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import vision_model as vm
import normalization_experiment as ne


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

NORMALIZATION_METHOD = "Percentile"
THRESHOLD_PERCENTILE = 80

FEATURE_INDICES = list(range(5, 103))


# ============================================================
# SCORE STATISTICS
# ============================================================

def calculate_statistics(scores, category, group):

    scores = np.asarray(scores)

    return {
        "Group": group,
        "Category": category,
        "Count": len(scores),
        "Mean": np.mean(scores),
        "Median": np.median(scores),
        "Q1": np.percentile(scores, 25),
        "Q3": np.percentile(scores, 75),
        "Minimum": np.min(scores),
        "Maximum": np.max(scores),
        "Std": np.std(scores),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n========== ANOMALY SCORE DISTRIBUTION ==========")

    training_paths = vm.collect_training_images()

    testing_paths, test_labels, categories = (
        vm.collect_testing_images()
    )

    test_labels = np.asarray(test_labels)
    categories = np.asarray(categories)

    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE,
    )

    print("\nTraining fit:", len(fit_paths))
    print("Calibration:", len(calibration_paths))
    print("Testing:", len(testing_paths))

    # Extract features
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

    # Train temporary model
    model = vm.train_anomaly_detector(X_fit)

    calibration_scores = vm.calculate_scores(
        model,
        X_calibration,
    )

    test_scores = vm.calculate_scores(
        model,
        X_test,
    )

    threshold = np.percentile(
        calibration_scores,
        THRESHOLD_PERCENTILE,
    )

    print("\n========== MODEL SETTINGS ==========")
    print("Normalization:", NORMALIZATION_METHOD)
    print("Selected features:", len(FEATURE_INDICES))
    print("Threshold percentile:", THRESHOLD_PERCENTILE)
    print(f"Threshold: {threshold:.6f}")

    # Collect score statistics
    results = []

    # Calibration normal images
    results.append(
        calculate_statistics(
            calibration_scores,
            "Normal calibration",
            "Calibration",
        )
    )

    # Test normal images
    normal_mask = test_labels == 0

    results.append(
        calculate_statistics(
            test_scores[normal_mask],
            "Good",
            "Testing",
        )
    )

    # Individual defect categories
    for category in sorted(set(categories)):

        mask = (
            (categories == category) &
            (test_labels == 1)
        )

        if np.sum(mask) == 0:
            continue

        results.append(
            calculate_statistics(
                test_scores[mask],
                category,
                "Testing",
            )
        )

    results_df = pd.DataFrame(results)

    print("\n========== SCORE DISTRIBUTION TABLE ==========")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # Detection statistics by category
    print("\n========== CATEGORY THRESHOLD ANALYSIS ==========")

    detection_results = []

    for category in sorted(set(categories)):

        mask = (
            (categories == category) &
            (test_labels == 1)
        )

        scores = test_scores[mask]

        if len(scores) == 0:
            continue

        detected = int(np.sum(scores >= threshold))
        missed = int(np.sum(scores < threshold))

        detection_results.append({
            "Category": category,
            "Total": len(scores),
            "Detected": detected,
            "Missed": missed,
            "Detection Rate (%)": detected / len(scores) * 100,
            "Median Score": np.median(scores),
            "Threshold": threshold,
        })

    detection_df = pd.DataFrame(detection_results)

    print(
        detection_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # Normal vs defect score overlap
    normal_scores = test_scores[normal_mask]
    defect_scores = test_scores[test_labels == 1]

    print("\n========== NORMAL VS DEFECT COMPARISON ==========")

    print("Normal test median:", np.median(normal_scores))
    print("Defect test median:", np.median(defect_scores))
    print("Normal test Q3:", np.percentile(normal_scores, 75))
    print("Defect test Q1:", np.percentile(defect_scores, 25))

    print(
        "Normal images above threshold:",
        np.sum(normal_scores >= threshold),
        "/",
        len(normal_scores),
    )

    print(
        "Defective images below threshold:",
        np.sum(defect_scores < threshold),
        "/",
        len(defect_scores),
    )

    print("\n========== EXPERIMENT COMPLETE ==========")
    print("No production model files were modified.")


if __name__ == "__main__":
    main()
