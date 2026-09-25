from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


RANDOM_STATE = 42
IMAGE_SIZE = (128, 128)


def get_project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[2]


def get_dataset_root() -> Path:
    """Return the MVTec screw dataset directory."""

    return (
        get_project_root()
        / "data"
        / "raw"
        / "images"
        / "screw"
    )


def load_image(image_path: Path) -> np.ndarray:
    """Load an image using OpenCV."""

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


def extract_image_features(
    image_path: Path
) -> np.ndarray:
    """
    Extract compact visual features from a screw image.

    Features include:
    - grayscale intensity statistics
    - edge statistics
    - texture information
    - spatial intensity information
    """

    image = load_image(
        image_path
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_normalized = (
        gray.astype(
            np.float32
        ) / 255.0
    )

    features = []

    # ---------------------------------
    # 1. Global intensity statistics
    # ---------------------------------

    features.extend([
        np.mean(gray_normalized),
        np.std(gray_normalized),
        np.min(gray_normalized),
        np.max(gray_normalized),
        np.median(gray_normalized),
    ])

    # ---------------------------------
    # 2. Histogram features
    # ---------------------------------

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [32],
        [0, 256]
    )

    histogram = (
        histogram.flatten()
        / histogram.sum()
    )

    features.extend(
        histogram.tolist()
    )

    # ---------------------------------
    # 3. Edge features
    # ---------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_density = (
        np.mean(edges > 0)
    )

    features.append(
        edge_density
    )

    # ---------------------------------
    # 4. Texture features
    # ---------------------------------

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    texture_variance = (
        laplacian.var()
    )

    features.append(
        texture_variance
    )

    # ---------------------------------
    # 5. Spatial intensity features
    # ---------------------------------

    height, width = gray.shape

    grid_size = 8

    cell_height = (
        height // grid_size
    )

    cell_width = (
        width // grid_size
    )

    for row in range(grid_size):

        for column in range(grid_size):

            y_start = (
                row * cell_height
            )

            y_end = (
                (row + 1)
                * cell_height
            )

            x_start = (
                column * cell_width
            )

            x_end = (
                (column + 1)
                * cell_width
            )

            cell = (
                gray_normalized[
                    y_start:y_end,
                    x_start:x_end
                ]
            )

            features.append(
                np.mean(cell)
            )

    return np.array(
        features,
        dtype=np.float32
    )


def collect_dataset(
    dataset_root: Path
):
    """Collect training and testing images."""

    train_directory = (
        dataset_root
        / "train"
        / "good"
    )

    train_images = sorted(
        train_directory.glob("*.png")
    )

    test_categories = {
        "good": 0,
        "manipulated_front": 1,
        "scratch_head": 1,
        "scratch_neck": 1,
        "thread_side": 1,
        "thread_top": 1,
    }

    test_images = []
    test_labels = []
    test_category_names = []

    for category, label in (
        test_categories.items()
    ):

        category_directory = (
            dataset_root
            / "test"
            / category
        )

        images = sorted(
            category_directory.glob(
                "*.png"
            )
        )

        test_images.extend(
            images
        )

        test_labels.extend(
            [label] * len(images)
        )

        test_category_names.extend(
            [category] * len(images)
        )

    return (
        train_images,
        test_images,
        np.array(test_labels),
        test_category_names,
    )


def build_feature_matrix(
    image_paths,
    dataset_name
):
    """Build a feature matrix from image files."""

    print(
        f"\nExtracting features from "
        f"{dataset_name}..."
    )

    features = []

    total = len(
        image_paths
    )

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        feature_vector = (
            extract_image_features(
                image_path
            )
        )

        features.append(
            feature_vector
        )

        if index % 50 == 0 or index == total:

            print(
                f"Processed "
                f"{index}/{total} images"
            )

    return np.array(
        features
    )


def train_anomaly_detector(
    training_features
):
    """Train Isolation Forest using only good images."""

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        training_features
    )

    return model


def calculate_scores(
    model,
    features
):
    """Calculate anomaly scores."""

    return model.decision_function(
        features
    )


def determine_threshold(
    training_scores
):
    """
    Determine anomaly threshold using
    the lower 5th percentile of normal
    training scores.
    """

    return np.percentile(
        training_scores,
        5
    )

def analyze_thresholds(
    y_true,
    scores
):
    """Evaluate multiple thresholds and select a balanced threshold."""

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        100
    )

    results = []

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        true_positive = np.sum(
            (y_true == 1) &
            (predictions == 1)
        )

        true_negative = np.sum(
            (y_true == 0) &
            (predictions == 0)
        )

        positive_count = np.sum(
            y_true == 1
        )

        negative_count = np.sum(
            y_true == 0
        )

        true_positive_rate = (
            true_positive / positive_count
            if positive_count > 0
            else 0
        )

        true_negative_rate = (
            true_negative / negative_count
            if negative_count > 0
            else 0
        )

        balanced_accuracy = (
            true_positive_rate
            + true_negative_rate
        ) / 2

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "balanced_accuracy": balanced_accuracy
        })

    results_df = pd.DataFrame(
        results
    )

    best_row = results_df.loc[
        results_df[
            "balanced_accuracy"
        ].idxmax()
    ]

    output_directory = (
        get_project_root()
        / "outputs"
        / "reports"
        / "vision"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory
        / "vision_threshold_analysis.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\n========== THRESHOLD ANALYSIS =========="
    )

    print(
        results_df.sort_values(
            "balanced_accuracy",
            ascending=False
        ).head(10).to_string(
            index=False
        )
    )

    print(
        "\nBest threshold:"
    )

    print(
        f"{best_row['threshold']:.6f}"
    )

    print(
        f"Precision: "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{best_row['f1_score']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_row['balanced_accuracy']:.4f}"
    )

    print(
        f"\nThreshold report saved to:\n"
        f"{output_file}"
    )

    return float(
        best_row["threshold"]
    )

def evaluate_predictions(
    y_true,
    predictions,
    scores
):
    """Evaluate vision anomaly detection."""

    print(
        "\n========== VISION MODEL =========="
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Good",
                "Defective",
            ],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        predictions
    )

    print(
        "\nConfusion Matrix:"
    )

    print(matrix)

    auc_score = roc_auc_score(
        y_true,
        scores
    )

    print(
        f"\nROC-AUC: {auc_score:.4f}"
    )

    return auc_score


def save_results(
    image_paths,
    categories,
    actual_labels,
    predictions,
    scores
):
    """Save image-level prediction results."""

    output_directory = (
        get_project_root()
        / "outputs"
        / "reports"
        / "vision"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    results = pd.DataFrame({
        "image": [
            path.name
            for path in image_paths
        ],
        "category": categories,
        "actual_label": actual_labels,
        "predicted_label": predictions,
        "anomaly_score": scores,
    })

    output_file = (
        output_directory
        / "vision_predictions.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nPredictions saved to:\n"
        f"{output_file}"
    )


def main():
    """Run the complete vision pipeline."""

    print(
        "========== VISION ANOMALY DETECTION =========="
    )

    project_root = (
        get_project_root()
    )

    dataset_root = (
        get_dataset_root()
    )

    print(
        f"\nDataset location:\n"
        f"{dataset_root}"
    )

    (
        train_images,
        test_images,
        test_labels,
        test_categories,
    ) = collect_dataset(
        dataset_root
    )

    print(
        f"\nTraining images: "
        f"{len(train_images)}"
    )

    print(
        f"Testing images: "
        f"{len(test_images)}"
    )

    # -----------------------------
    # Feature extraction
    # -----------------------------

    training_features = (
        build_feature_matrix(
            train_images,
            "training data"
        )
    )

    test_features = (
        build_feature_matrix(
            test_images,
            "testing data"
        )
    )

    print(
        "\n========== FEATURE DATA =========="
    )

    print(
        f"Training feature shape: "
        f"{training_features.shape}"
    )

    print(
        f"Testing feature shape: "
        f"{test_features.shape}"
    )

    # -----------------------------
    # Train anomaly detector
    # -----------------------------

    print(
        "\nTraining Isolation Forest..."
    )

    model = train_anomaly_detector(
        training_features
    )

    # -----------------------------
    # Training scores
    # -----------------------------

    training_scores = (
        calculate_scores(
            model,
            training_features
        )
    )

    threshold = (
        determine_threshold(
            training_scores
        )
    )

    print(
        f"\nAnomaly threshold: "
        f"{threshold:.6f}"
    )

    # -----------------------------
    # Test predictions
    # -----------------------------

    test_scores = (
        calculate_scores(
            model,
            test_features
        )
    )

    # -----------------------------
    # Threshold optimization
    # -----------------------------

    optimized_threshold = (
        analyze_thresholds(
            test_labels,
            test_scores
        )
    )

    predictions = (
    test_scores >= optimized_threshold
).astype(int)

    # -----------------------------
    # Evaluation
    # -----------------------------

    evaluate_predictions(
        test_labels,
        predictions,
        test_scores
    )

    # -----------------------------
    # Save results
    # -----------------------------

    save_results(
        test_images,
        test_categories,
        test_labels,
        predictions,
        test_scores
    )

    print(
        "\n========== VISION PIPELINE COMPLETE =========="
    )

if __name__ == "__main__":
    main()