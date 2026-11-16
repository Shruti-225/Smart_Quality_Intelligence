
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "images" / "screw"

TRAIN_GOOD_DIR = DATA_DIR / "train" / "good"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = ROOT / "outputs" / "reports" / "vision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)

WINDOW_SIZES = [3, 5, 9, 15]

ROI_TYPES = {
    "Full Image": 0.0,
    "Central 80 Percent": 0.10,
}

RANDOM_STATE = 42


# ============================================================
# IMAGE DISCOVERY
# ============================================================

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_images(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def load_normalized_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, low, high)

    image = (
        (image - low) / (high - low)
    ) * 255.0

    return image.astype(np.float32)


# ============================================================
# LOCAL CONTRAST CALCULATION
# ============================================================

def calculate_local_contrast(image, window_size):
    """
    Calculate local standard deviation using a box filter.
    This represents local contrast at a particular scale.
    """

    mean = cv2.blur(
        image,
        (window_size, window_size),
        borderType=cv2.BORDER_REFLECT
    )

    mean_squared = cv2.blur(
        image ** 2,
        (window_size, window_size),
        borderType=cv2.BORDER_REFLECT
    )

    variance = np.maximum(
        mean_squared - mean ** 2,
        0
    )

    return np.sqrt(variance)


# ============================================================
# REGION EXTRACTION
# ============================================================

def select_region(image, border_fraction):
    if border_fraction == 0:
        return image

    height, width = image.shape

    border_y = int(height * border_fraction)
    border_x = int(width * border_fraction)

    return image[
        border_y:height - border_y,
        border_x:width - border_x
    ]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path, group):
    image = load_normalized_image(image_path)

    records = []

    for roi_name, border_fraction in ROI_TYPES.items():

        region = select_region(image, border_fraction)

        for window_size in WINDOW_SIZES:

            contrast_map = calculate_local_contrast(
                region,
                window_size
            )

            records.append({
                "Image": image_path.name,
                "Group": group,
                "ROI": roi_name,
                "Window": window_size,
                "contrast_mean": float(
                    np.mean(contrast_map)
                ),
                "contrast_std": float(
                    np.std(contrast_map)
                ),
                "contrast_p90": float(
                    np.percentile(contrast_map, 90)
                ),
                "contrast_max": float(
                    np.max(contrast_map)
                ),
            })

    return records


# ============================================================
# DATASET COLLECTION
# ============================================================

print("\n========== MULTI-SCALE LOCAL CONTRAST ANALYSIS ==========\n")

train_paths = get_images(TRAIN_GOOD_DIR)

if len(train_paths) != 320:
    raise ValueError(
        f"Expected 320 training Good images, found {len(train_paths)}"
    )

fit_paths, calibration_paths = train_test_split(
    train_paths,
    test_size=0.20,
    random_state=RANDOM_STATE
)

test_good_paths = get_images(TEST_DIR / "good")

defect_categories = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]

category_paths = {}

for category in defect_categories:
    category_paths[category] = get_images(
        TEST_DIR / category
    )

print("Training Fit Good:", len(fit_paths))
print("Calibration Good:", len(calibration_paths))
print("Test Good:", len(test_good_paths))

for category, paths in category_paths.items():
    print(f"{category}: {len(paths)}")


# ============================================================
# EXTRACT ALL FEATURES
# ============================================================

all_records = []

datasets = [
    ("Training Fit Good", fit_paths),
    ("Calibration Good", calibration_paths),
    ("Test Good", test_good_paths),
]

for category, paths in category_paths.items():
    datasets.append((category, paths))


for group, paths in datasets:

    print(f"\nProcessing {group}...")

    for index, path in enumerate(paths, start=1):

        try:
            records = extract_features(path, group)
            all_records.extend(records)

        except Exception as error:
            print(f"Skipped {path.name}: {error}")

        if index % 50 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)}")


# ============================================================
# SAVE IMAGE-LEVEL FEATURE VALUES
# ============================================================

features_df = pd.DataFrame(all_records)

if features_df.empty:
    raise RuntimeError("No image features were extracted.")

feature_path = (
    OUTPUT_DIR / "multiscale_contrast_feature_values.csv"
)

features_df.to_csv(feature_path, index=False)


# ============================================================
# FEATURE DISTRIBUTIONS
# ============================================================

print("\n========== FEATURE DISTRIBUTIONS ==========\n")

summary = (
    features_df
    .groupby(["Group", "ROI", "Window"])[
        [
            "contrast_mean",
            "contrast_std",
            "contrast_p90",
            "contrast_max",
        ]
    ]
    .agg(["mean", "median", "std"])
    .round(4)
)

print(summary.to_string())

summary_path = (
    OUTPUT_DIR / "multiscale_contrast_distributions.csv"
)

summary.to_csv(summary_path)


# ============================================================
# TRAINING GOOD VS TEST GOOD SHIFT
# ============================================================

print("\n========== TRAINING GOOD VS TEST GOOD ==========\n")

shift_results = []

for roi in ROI_TYPES:
    for window in WINDOW_SIZES:
        for feature in [
            "contrast_mean",
            "contrast_std",
            "contrast_p90",
        ]:

            train_values = features_df[
                (features_df["Group"] == "Training Fit Good")
                & (features_df["ROI"] == roi)
                & (features_df["Window"] == window)
            ][feature].to_numpy()

            test_values = features_df[
                (features_df["Group"] == "Test Good")
                & (features_df["ROI"] == roi)
                & (features_df["Window"] == window)
            ][feature].to_numpy()

            labels = np.concatenate([
                np.zeros(len(train_values)),
                np.ones(len(test_values))
            ])

            values = np.concatenate([
                train_values,
                test_values
            ])

            auc = roc_auc_score(labels, values)

            shift_results.append({
                "ROI": roi,
                "Window": window,
                "Feature": feature,
                "Training Mean": np.mean(train_values),
                "Test Good Mean": np.mean(test_values),
                "Train_vs_Test_AUC": auc,
            })

            print(
                f"{roi} | Window {window} | {feature}: "
                f"AUC={auc:.4f}"
            )


shift_df = pd.DataFrame(shift_results)

shift_df.to_csv(
    OUTPUT_DIR / "multiscale_contrast_train_test_shift.csv",
    index=False
)


# ============================================================
# TEST GOOD VS DEFECT CATEGORY ANALYSIS
# ============================================================

print("\n========== TEST GOOD VS DEFECT CATEGORIES ==========\n")

category_results = []

for roi in ROI_TYPES:
    for window in WINDOW_SIZES:
        for feature in [
            "contrast_mean",
            "contrast_std",
            "contrast_p90",
        ]:

            good_values = features_df[
                (features_df["Group"] == "Test Good")
                & (features_df["ROI"] == roi)
                & (features_df["Window"] == window)
            ][feature].to_numpy()

            for category in defect_categories:

                defect_values = features_df[
                    (features_df["Group"] == category)
                    & (features_df["ROI"] == roi)
                    & (features_df["Window"] == window)
                ][feature].to_numpy()

                labels = np.concatenate([
                    np.zeros(len(good_values)),
                    np.ones(len(defect_values))
                ])

                values = np.concatenate([
                    good_values,
                    defect_values
                ])

                auc = roc_auc_score(labels, values)

                category_results.append({
                    "ROI": roi,
                    "Window": window,
                    "Feature": feature,
                    "Category": category,
                    "AUC": auc,
                })

                print(
                    f"{roi} | Window {window} | "
                    f"{feature} | {category}: AUC={auc:.4f}"
                )


category_df = pd.DataFrame(category_results)

category_df.to_csv(
    OUTPUT_DIR / "multiscale_contrast_category_analysis.csv",
    index=False
)


# ============================================================
# BEST RESULTS SUMMARY
# ============================================================

print("\n========== TOP CATEGORY RESULTS ==========\n")

top_results = category_df.sort_values(
    "AUC",
    ascending=False
).head(20)

print(top_results.round(4).to_string(index=False))


print("\n========== REPORTS SAVED ==========\n")

print(feature_path)
print(summary_path)

print(
    OUTPUT_DIR / "multiscale_contrast_train_test_shift.csv"
)

print(
    OUTPUT_DIR / "multiscale_contrast_category_analysis.csv"
)

print("\nEXPERIMENT COMPLETE")
print("No production model or saved model artifacts were changed.")
