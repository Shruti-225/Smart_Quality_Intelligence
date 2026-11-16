
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import vision_model as vm
import normalization_experiment as ne


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 128
NORMALIZATION = "Percentile"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "vision"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# IMAGE LOADING
# --------------------------------------------------

def load_image(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = ne.normalize_image(
        image,
        NORMALIZATION
    )

    return np.asarray(image, dtype=np.uint8)


# --------------------------------------------------
# LOCAL BINARY PATTERN
# --------------------------------------------------

def calculate_lbp(image):

    image = image.astype(np.int16)

    center = image[1:-1, 1:-1]

    neighbors = [
        image[:-2, :-2],
        image[:-2, 1:-1],
        image[:-2, 2:],
        image[1:-1, 2:],
        image[2:, 2:],
        image[2:, 1:-1],
        image[2:, :-2],
        image[1:-1, :-2],
    ]

    lbp = np.zeros(
        center.shape,
        dtype=np.uint8
    )

    for bit, neighbor in enumerate(neighbors):

        lbp |= (
            (neighbor >= center).astype(np.uint8)
            << bit
        )

    histogram = np.histogram(
        lbp,
        bins=256,
        range=(0, 256),
        density=True
    )[0]

    entropy = -np.sum(
        histogram * np.log2(histogram + 1e-10)
    )

    return histogram, entropy


# --------------------------------------------------
# TEXTURE FEATURE EXTRACTION
# --------------------------------------------------

def extract_texture_features(image):

    features = {}

    # 1. LBP texture
    lbp_histogram, lbp_entropy = calculate_lbp(image)

    features["lbp_entropy"] = lbp_entropy

    features["lbp_uniformity"] = np.sum(
        lbp_histogram ** 2
    )

    # 2. Gradient magnitude
    image_float = image.astype(np.float32)

    sobel_x = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    features["gradient_mean"] = np.mean(gradient)
    features["gradient_std"] = np.std(gradient)
    features["gradient_p90"] = np.percentile(gradient, 90)

    # 3. Local contrast using local variance
    image_float = image.astype(np.float32)

    local_mean = cv2.blur(
        image_float,
        (7, 7)
    )

    local_mean_squared = cv2.blur(
        image_float ** 2,
        (7, 7)
    )

    local_variance = np.maximum(
        local_mean_squared - local_mean ** 2,
        0
    )

    local_std = np.sqrt(local_variance)

    features["local_contrast_mean"] = np.mean(local_std)
    features["local_contrast_std"] = np.std(local_std)
    features["local_contrast_p90"] = np.percentile(local_std, 90)

    # 4. Edge density
    edges = cv2.Canny(
        image,
        threshold1=50,
        threshold2=150
    )

    features["edge_density"] = np.mean(
        edges > 0
    )

    # Additional intensity-independent structural information
    features["edge_count"] = np.sum(edges > 0)

    return features


# --------------------------------------------------
# GROUP EXTRACTION
# --------------------------------------------------

def extract_group_features(paths, group_name):

    rows = []

    print(f"\nExtracting {group_name}...")

    for index, path in enumerate(paths, start=1):

        image = load_image(path)

        features = extract_texture_features(image)

        features["Group"] = group_name
        features["Image"] = Path(path).name

        rows.append(features)

        if index % 50 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)}")

    return pd.DataFrame(rows)


# --------------------------------------------------
# SEPARABILITY ANALYSIS
# --------------------------------------------------

FEATURE_NAMES = [
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
]


def calculate_separability(group_a, group_b, feature):

    values_a = group_a[feature].to_numpy()
    values_b = group_b[feature].to_numpy()

    labels = np.concatenate([
        np.zeros(len(values_a)),
        np.ones(len(values_b))
    ])

    values = np.concatenate([
        values_a,
        values_b
    ])

    auc = roc_auc_score(labels, values)

    return max(auc, 1 - auc)


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("\n========== LOCAL TEXTURE ANALYSIS ==========\n")

    training_paths = vm.collect_training_images()

    testing_paths, test_labels, test_categories = (
        vm.collect_testing_images()
    )

    # Preserve original 256 / 64 training split
    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE
    )

    test_paths = np.asarray(testing_paths)
    test_categories = np.asarray(test_categories)
    test_labels = np.asarray(test_labels)

    # Identify normal test images
    normal_mask = np.array([
        str(category).strip().lower() in {"good", "normal"}
        for category in test_categories
    ])

    if not np.any(normal_mask):

        normal_mask = np.array([
            str(label).strip().lower() in {"0", "good", "normal"}
            for label in test_labels
        ])

    defective_mask = ~normal_mask

    # Create groups
    groups = {
        "Training Fit Good": list(fit_paths),
        "Calibration Good": list(calibration_paths),
        "Test Good": list(test_paths[normal_mask]),
    }

    for category in sorted(set(test_categories[defective_mask])):

        mask = (
            (test_categories == category)
            & defective_mask
        )

        groups[str(category)] = list(test_paths[mask])

    print("\n========== IMAGE COUNTS ==========\n")

    for name, paths in groups.items():
        print(f"{name}: {len(paths)}")

    # Extract features
    frames = []

    for group_name, paths in groups.items():

        frame = extract_group_features(
            paths,
            group_name
        )

        frames.append(frame)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    # --------------------------------------------------
    # FEATURE DISTRIBUTIONS
    # --------------------------------------------------

    print("\n========== FEATURE DISTRIBUTIONS ==========\n")

    summary = (
        data.groupby("Group")[FEATURE_NAMES]
        .agg(["mean", "median", "std"])
        .round(5)
    )

    print(summary.to_string())

    # --------------------------------------------------
    # TRAINING GOOD VS TEST GOOD
    # --------------------------------------------------

    print("\n========== TRAINING GOOD VS TEST GOOD ==========\n")

    training_good = data[
        data["Group"] == "Training Fit Good"
    ]

    test_good = data[
        data["Group"] == "Test Good"
    ]

    shift_rows = []

    for feature in FEATURE_NAMES:

        auc = calculate_separability(
            training_good,
            test_good,
            feature
        )

        row = {
            "Feature": feature,
            "Training Mean": training_good[feature].mean(),
            "Test Good Mean": test_good[feature].mean(),
            "Separability AUC": auc,
        }

        shift_rows.append(row)

        print(
            f"{feature}: "
            f"Training={row['Training Mean']:.5f}, "
            f"Test={row['Test Good Mean']:.5f}, "
            f"AUC={auc:.5f}"
        )

    # --------------------------------------------------
    # TEST GOOD VS DEFECT CATEGORIES
    # --------------------------------------------------

    print("\n========== TEST GOOD VS DEFECT CATEGORIES ==========\n")

    category_rows = []

    for category in sorted(set(test_categories[defective_mask])):

        category_data = data[
            data["Group"] == str(category)
        ]

        print(f"\n--- {category} ---")

        for feature in FEATURE_NAMES:

            auc = calculate_separability(
                test_good,
                category_data,
                feature
            )

            category_rows.append({
                "Category": str(category),
                "Feature": feature,
                "Separability AUC": auc,
                "Good Mean": test_good[feature].mean(),
                "Defect Mean": category_data[feature].mean(),
            })

            print(
                f"{feature}: "
                f"AUC={auc:.5f}"
            )

    # --------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------

    data_path = (
        OUTPUT_DIR
        / "local_texture_feature_values.csv"
    )

    shift_path = (
        OUTPUT_DIR
        / "local_texture_train_test_shift.csv"
    )

    category_path = (
        OUTPUT_DIR
        / "local_texture_category_analysis.csv"
    )

    data.to_csv(
        data_path,
        index=False
    )

    pd.DataFrame(shift_rows).to_csv(
        shift_path,
        index=False
    )

    pd.DataFrame(category_rows).to_csv(
        category_path,
        index=False
    )

    print("\n========== REPORTS SAVED ==========\n")

    print(data_path)
    print(shift_path)
    print(category_path)

    print("\n========== EXPERIMENT COMPLETE ==========\n")
    print("No production model or saved model artifacts were changed.")


if __name__ == "__main__":
    main()
