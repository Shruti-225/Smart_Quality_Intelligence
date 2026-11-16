
from pathlib import Path

import cv2
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


# ============================================================
# NORMALIZATION METHODS
# ============================================================

def normalize_image(gray, method):

    if method == "Original":
        return gray

    if method == "Percentile":

        low, high = np.percentile(gray, (1, 99))

        if high <= low:
            return gray

        clipped = np.clip(gray, low, high)

        return cv2.normalize(
            clipped,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    if method == "Histogram Equalization":
        return cv2.equalizeHist(gray)

    if method == "CLAHE":

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        return clahe.apply(gray)

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_normalized_features(image_path, method):

    image = vm.load_image(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = normalize_image(gray, method)

    features = []

    # Intensity statistics
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray),
        np.median(gray),
    ])

    # Histogram
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [32],
        [0, 256],
    )

    histogram = cv2.normalize(
        histogram,
        histogram,
    )

    features.extend(histogram.flatten())

    # Edge density
    edges = cv2.Canny(gray, 50, 150)

    features.append(np.mean(edges > 0))

    # Laplacian variance
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    features.append(laplacian.var())

    # Spatial features
    height, width = gray.shape

    cell_height = height // 8
    cell_width = width // 8

    for row in range(8):
        for col in range(8):

            cell = gray[
                row * cell_height:(row + 1) * cell_height,
                col * cell_width:(col + 1) * cell_width,
            ]

            features.append(np.mean(cell))

    result = np.asarray(features, dtype=np.float64)

    if len(result) != 103:
        raise ValueError("Expected 103 features.")

    return result


def build_normalized_matrix(paths, method, description):

    features = []

    print(f"\nExtracting {description}...")

    for index, path in enumerate(paths, start=1):

        features.append(
            extract_normalized_features(path, method)
        )

        if index % 50 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)}")

    return np.asarray(features, dtype=np.float64)


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(
    method,
    fit_paths,
    calibration_paths,
    testing_paths,
    test_labels,
):

    print("\n" + "=" * 55)
    print(f"METHOD: {method}")
    print("=" * 55)

    fit_features = build_normalized_matrix(
        fit_paths,
        method,
        "training images",
    )

    calibration_features = build_normalized_matrix(
        calibration_paths,
        method,
        "calibration images",
    )

    testing_features = build_normalized_matrix(
        testing_paths,
        method,
        "testing images",
    )

    print("\nTraining temporary Isolation Forest...")

    model = vm.train_anomaly_detector(fit_features)

    calibration_scores = vm.calculate_scores(
        model,
        calibration_features,
    )

    threshold = vm.determine_threshold(
        calibration_scores
    )

    test_scores = vm.calculate_scores(
        model,
        testing_features,
    )

    print("\n--- SCORE DIRECTION DIAGNOSTICS ---")

    print(
    "Calibration Good median:",
    np.median(calibration_scores),
    )

    print(
    "Test Good median:",
    np.median(test_scores[test_labels == 0]),
    )

    print(
    "Test Defect median:",
    np.median(test_scores[test_labels == 1]),
    )

    print(
    "AUC (higher score = defect):",
    roc_auc_score(test_labels, test_scores),
    )

    print(
    "AUC (lower score = defect):",
    roc_auc_score(test_labels, -test_scores),
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
    )

    print("\n========== RESULTS ==========")

    print(f"Threshold: {threshold:.6f}")
    print(f"ROC-AUC: {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
    print(f"Balanced Accuracy: {balanced_acc:.6f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "Method": method,
        "Threshold": threshold,
        "ROC_AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Balanced_Accuracy": balanced_acc,
        "True_Negatives": int(matrix[0, 0]),
        "False_Positives": int(matrix[0, 1]),
        "False_Negatives": int(matrix[1, 0]),
        "True_Positives": int(matrix[1, 1]),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========== NORMALIZATION EXPERIMENT ==========")

    training_paths = vm.collect_training_images()

    testing_paths, test_labels, test_categories = (
        vm.collect_testing_images()
    )

    # Preserve the exact original 80/20 split.
    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE,
    )

    print(f"\nTraining fit images: {len(fit_paths)}")
    print(f"Calibration images: {len(calibration_paths)}")
    print(f"Testing images: {len(testing_paths)}")

    methods = [
        "Original",
        "Percentile",
        "Histogram Equalization",
        "CLAHE",
    ]

    results = []

    for method in methods:

        result = run_experiment(
            method,
            fit_paths,
            calibration_paths,
            testing_paths,
            test_labels,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n========== FINAL COMPARISON ==========")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n========== EXPERIMENT COMPLETE ==========")

    print(
        "\nNo existing model or project artifacts "
        "were overwritten."
    )


if __name__ == "__main__":
    main()
