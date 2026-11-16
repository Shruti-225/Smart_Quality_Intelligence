
from pathlib import Path
import json

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
IMAGE_SIZE = (128, 128)
CALIBRATION_PERCENTILE = 95

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT_DIR / "data" / "raw" / "images" / "screw"

MODEL_DIR = ROOT_DIR / "outputs" / "models"
REPORT_DIR = ROOT_DIR / "outputs" / "reports" / "vision"

MODEL_PATH = MODEL_DIR / "vision_isolation_forest.pkl"
METADATA_PATH = MODEL_DIR / "vision_model_metadata.json"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)

    return image


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_image_features(image_path):
    """
    Extract 103 features from an image.

    Features:
    1. Five intensity statistics
    2. Thirty-two histogram features
    3. Edge density
    4. Laplacian variance
    5. Sixty-four spatial cell means
    """

    image = load_image(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    # 1. Intensity statistics: 5 features
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray),
        np.median(gray),
    ])

    # 2. Histogram: 32 features
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

    # 3. Edge density: 1 feature
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.mean(edges > 0)

    features.append(edge_density)

    # 4. Laplacian variance: 1 feature
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    features.append(laplacian.var())

    # 5. Spatial cell means: 64 features
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
        raise ValueError(
            f"Expected 103 features, got {len(result)}"
        )

    return result


# ============================================================
# DATASET COLLECTION
# ============================================================

def collect_training_images():
    training_dir = DATASET_DIR / "train" / "good"

    if not training_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {training_dir}"
        )

    image_paths = sorted([
        path
        for path in training_dir.iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    if len(image_paths) < 2:
        raise ValueError(
            "At least two good training images are required."
        )

    return image_paths


def collect_testing_images():
    testing_dir = DATASET_DIR / "test"

    if not testing_dir.exists():
        raise FileNotFoundError(
            f"Testing directory not found: {testing_dir}"
        )

    image_paths = []
    labels = []
    categories = []

    category_folders = sorted([
        folder
        for folder in testing_dir.iterdir()
        if folder.is_dir()
    ])

    for folder in category_folders:

        category = folder.name

        if category.lower() == "good":
            label = 0
        else:
            label = 1

        for image_path in sorted(folder.iterdir()):

            if image_path.suffix.lower() not in [
                ".png", ".jpg", ".jpeg"
            ]:
                continue

            image_paths.append(image_path)
            labels.append(label)
            categories.append(category)

    if not image_paths:
        raise ValueError("No testing images were found.")

    return image_paths, np.asarray(labels), categories


# ============================================================
# FEATURE MATRIX CREATION
# ============================================================

def build_feature_matrix(image_paths, description):

    features = []

    print(f"\nExtracting features from {description}...")

    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):

        try:
            feature_vector = extract_image_features(image_path)
            features.append(feature_vector)

        except Exception as error:
            raise RuntimeError(
                f"Feature extraction failed for {image_path}: {error}"
            ) from error

        if index % 50 == 0 or index == total:
            print(f"Processed {index}/{total} images")

    return np.asarray(features, dtype=np.float64)


# ============================================================
# MODEL TRAINING
# ============================================================

def train_anomaly_detector(training_features):

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(training_features)

    return model


# ============================================================
# ANOMALY SCORE CALCULATION
# ============================================================

def calculate_scores(model, features):
    """
    IsolationForest decision_function:
        Higher = more normal
        Lower  = more anomalous

    We reverse the sign so that:
        Higher anomaly score = more anomalous
        Lower anomaly score  = more normal
    """

    return -model.decision_function(features)


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def determine_threshold(normal_scores):

    threshold = np.percentile(
        normal_scores,
        CALIBRATION_PERCENTILE,
    )

    return float(threshold)


# ============================================================
# SAVE MODEL AND METADATA
# ============================================================

def save_model_artifacts(model, threshold):

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "threshold": float(threshold),
        "image_size": list(IMAGE_SIZE),
        "random_state": RANDOM_STATE,
        "feature_count": 103,
        "score_type": "negative_decision_function",
        "prediction_rule": "anomaly_score >= threshold",
        "calibration_percentile": CALIBRATION_PERCENTILE,
        "dataset": "MVTec AD Screw",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_predictions(true_labels, predictions, anomaly_scores):

    print("\n========== VISION MODEL ==========")

    print("\nAverage anomaly scores:")
    print(
        "Good images:",
        round(float(np.mean(anomaly_scores[true_labels == 0])), 6)
    )
    print(
        "Defective images:",
        round(float(np.mean(anomaly_scores[true_labels == 1])), 6)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            target_names=["Good", "Defective"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    )

    print("Confusion Matrix:")
    print(matrix)

    if len(np.unique(true_labels)) == 2:

        auc = roc_auc_score(
            true_labels,
            anomaly_scores,
        )

        print(f"\nROC-AUC: {auc:.4f}")

    else:
        auc = None
        print("\nROC-AUC unavailable: only one class found.")

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy_score(
                true_labels,
                predictions,
            ),
            4,
        ),
    )

    print(
        "Precision:",
        round(
            precision_score(
                true_labels,
                predictions,
                zero_division=0,
            ),
            4,
        ),
    )

    print(
        "Recall:",
        round(
            recall_score(
                true_labels,
                predictions,
                zero_division=0,
            ),
            4,
        ),
    )

    print(
        "F1 Score:",
        round(
            f1_score(
                true_labels,
                predictions,
                zero_division=0,
            ),
            4,
        ),
    )

    return auc, matrix


# ============================================================
# SAVE PREDICTION REPORT
# ============================================================

def save_prediction_report(
    image_paths,
    categories,
    true_labels,
    predictions,
    anomaly_scores,
):

    report = pd.DataFrame({
        "image": [path.name for path in image_paths],
        "category": categories,
        "actual_label": [
            "Defective" if label == 1 else "Good"
            for label in true_labels
        ],
        "predicted_label": [
            "Defective" if prediction == 1 else "Good"
            for prediction in predictions
        ],
        "anomaly_score": anomaly_scores,
    })

    output_path = REPORT_DIR / "vision_predictions.csv"

    report.to_csv(output_path, index=False)

    print(f"\nPrediction report saved: {output_path}")

    return report


# ============================================================
# SAVE CATEGORY PERFORMANCE
# ============================================================

def save_category_performance(report):

    results = []

    for category in sorted(report["category"].unique()):

        category_data = report[
            report["category"] == category
        ]

        correct = (
            category_data["actual_label"]
            == category_data["predicted_label"]
        ).sum()

        total = len(category_data)

        results.append({
            "category": category,
            "total_images": total,
            "correctly_classified": int(correct),
            "incorrectly_classified": int(total - correct),
            "detection_rate": float(correct / total),
            "average_anomaly_score": float(
                category_data["anomaly_score"].mean()
            ),
            "minimum_anomaly_score": float(
                category_data["anomaly_score"].min()
            ),
            "maximum_anomaly_score": float(
                category_data["anomaly_score"].max()
            ),
        })

    result_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "defect_performance.csv"

    result_df.to_csv(output_path, index=False)

    print(f"\nCategory report saved: {output_path}")
    print("\nCategory Performance:")
    print(result_df.to_string(index=False))

    return result_df


# ============================================================
# SAVE DEFECT TYPE SUMMARY
# ============================================================

def save_defect_type_summary(categories):

    summary = (
        pd.Series(categories)
        .value_counts()
        .rename_axis("defect_type")
        .reset_index(name="image_count")
    )

    summary["folder"] = summary["defect_type"]

    summary = summary[
        ["defect_type", "folder", "image_count"]
    ]

    output_path = REPORT_DIR / "defect_type_summary.csv"

    summary.to_csv(output_path, index=False)

    print(f"\nDefect type summary saved: {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n========== VISION ANOMALY DETECTION ==========")

    print(f"\nDataset location:\n{DATASET_DIR}")

    create_directories()

    # --------------------------------------------------------
    # 1. Collect images
    # --------------------------------------------------------

    training_paths = collect_training_images()

    testing_paths, test_labels, test_categories = (
        collect_testing_images()
    )

    print(f"\nTraining images: {len(training_paths)}")
    print(f"Testing images: {len(testing_paths)}")

    # --------------------------------------------------------
    # 2. Extract features
    # --------------------------------------------------------

    training_features = build_feature_matrix(
        training_paths,
        "training data",
    )

    testing_features = build_feature_matrix(
        testing_paths,
        "testing data",
    )

    print("\n========== FEATURE DATA ==========")

    print("Training feature shape:", training_features.shape)
    print("Testing feature shape:", testing_features.shape)

    if training_features.shape[1] != 103:
        raise ValueError("Training features must contain 103 columns.")

    if testing_features.shape[1] != 103:
        raise ValueError("Testing features must contain 103 columns.")

    # --------------------------------------------------------
    # 3. Split normal training data
    # --------------------------------------------------------

    fit_features, calibration_features = train_test_split(
        training_features,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # 4. Train Isolation Forest
    # --------------------------------------------------------

    print("\nTraining Isolation Forest...")

    model = train_anomaly_detector(fit_features)

    # --------------------------------------------------------
    # 5. Calibrate using held-out normal images
    # --------------------------------------------------------

    calibration_scores = calculate_scores(
        model,
        calibration_features,
    )

    threshold = determine_threshold(calibration_scores)

    print(f"\nCalibrated threshold: {threshold:.6f}")

    # --------------------------------------------------------
    # 6. Save trained model
    # --------------------------------------------------------

    save_model_artifacts(model, threshold)

    # --------------------------------------------------------
    # 7. Evaluate on untouched test images
    # --------------------------------------------------------

    test_scores = calculate_scores(
        model,
        testing_features,
    )

    predictions = (
        test_scores >= threshold
    ).astype(int)

    # Diagnostic checks: confirm the score direction.
    good_scores = test_scores[test_labels == 0]
    defective_scores = test_scores[test_labels == 1]

    print("\n========== SCORE DIRECTION CHECK ==========")

    if len(good_scores):
        print(
            "Mean Good score:",
            round(float(np.mean(good_scores)), 6),
        )

    if len(defective_scores):
        print(
            "Mean Defective score:",
            round(float(np.mean(defective_scores)), 6),
        )

    # --------------------------------------------------------
    # 8. Evaluate model
    # --------------------------------------------------------

    evaluate_predictions(
        test_labels,
        predictions,
        test_scores,
    )

    # --------------------------------------------------------
    # 9. Save reports
    # --------------------------------------------------------

    report = save_prediction_report(
        testing_paths,
        test_categories,
        test_labels,
        predictions,
        test_scores,
    )

    save_category_performance(report)

    save_defect_type_summary(test_categories)

    print("\n========== VISION PIPELINE COMPLETE ==========")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
