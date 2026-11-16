
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "images" / "screw"
TEST_DIR = DATA_DIR / "test"
MASK_DIR = DATA_DIR / "ground_truth"

OUTPUT_DIR = ROOT / "outputs" / "reports" / "vision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]

WINDOW_SIZES = [9, 15]
TOP_PERCENTAGES = [5, 10, 20]

IMAGE_SIZE = (128, 128)
MASK_THRESHOLD = 0

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high > low:
        image = np.clip(image, low, high)
        image = (image - low) / (high - low) * 255.0
    else:
        image = np.zeros_like(image)

    return image.astype(np.float32)


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    return mask > MASK_THRESHOLD


# ============================================================
# LOCAL CONTRAST
# ============================================================

def local_contrast(image, window_size):

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
# ALIGNMENT METRICS
# ============================================================


def calculate_metrics(contrast_map, mask, category, image_name, window):

    mask = mask.astype(bool)

    defect_values = contrast_map[mask]
    background_values = contrast_map[~mask]

    if len(defect_values) == 0 or len(background_values) == 0:
        raise ValueError("Mask must contain defect and background pixels.")

    labels = mask.astype(np.uint8).ravel()
    scores = contrast_map.ravel()

    pixel_auc = roc_auc_score(labels, scores)

    contrast_ratio = (
        float(np.mean(defect_values)) /
        max(float(np.mean(background_values)), 1e-8)
    )

    results = []

    flat_scores = contrast_map.ravel()
    total_pixels = len(flat_scores)

    for percentage in TOP_PERCENTAGES:

        # Select exactly the requested number of highest-contrast pixels
        top_count = max(1, int(np.ceil(total_pixels * percentage / 100)))

        top_indices = np.argsort(
            flat_scores, kind="stable"
        )[-top_count:]

        highlighted_flat = np.zeros(total_pixels, dtype=bool)
        highlighted_flat[top_indices] = True

        highlighted = highlighted_flat.reshape(contrast_map.shape)

        true_positive = np.count_nonzero(highlighted & mask)
        highlighted_count = np.count_nonzero(highlighted)
        defect_count = np.count_nonzero(mask)

        coverage = true_positive / defect_count
        precision = true_positive / highlighted_count

        results.append({
            "Category": category,
            "Image": image_name,
            "Window": window,
            "Top Contrast Percent": percentage,
            "Pixel ROC-AUC": pixel_auc,
            "Defect Coverage": coverage,
            "Highlight Precision": precision,
            "Contrast Inside Defect": float(np.mean(defect_values)),
            "Contrast Outside Defect": float(np.mean(background_values)),
            "Contrast Ratio": contrast_ratio,
            "Defect Pixel Percent": 100 * defect_count / mask.size,
        })

    return results



# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("\n========== DEFECT MASK VS LOCAL CONTRAST ==========\n")

all_results = []
pair_summary = []

for category in CATEGORIES:

    image_dir = TEST_DIR / category
    mask_dir = MASK_DIR / category

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask folder: {mask_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    print(f"\n========== {category} ==========")

    matched_count = 0

    for image_path in image_paths:

        mask_path = mask_dir / f"{image_path.stem}_mask.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing matching mask for {image_path.name}: {mask_path}"
            )

        image = load_image(image_path)
        mask = load_mask(mask_path)

        if not np.any(mask):
            raise ValueError(f"Empty defect mask: {mask_path}")

        if np.all(mask):
            raise ValueError(f"Mask has no background pixels: {mask_path}")

        matched_count += 1

        for window in WINDOW_SIZES:

            contrast_map = local_contrast(image, window)

            results = calculate_metrics(
                contrast_map,
                mask,
                category,
                image_path.name,
                window
            )

            all_results.extend(results)

    print("Matched image-mask pairs:", matched_count)

    pair_summary.append({
        "Category": category,
        "Image Count": len(image_paths),
        "Matched Pairs": matched_count,
    })


# ============================================================
# SAVE IMAGE-LEVEL RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

if results_df.empty:
    raise RuntimeError("No alignment results were generated.")

results_path = (
    OUTPUT_DIR / "defect_mask_contrast_alignment.csv"
)

results_df.to_csv(results_path, index=False)


# ============================================================
# CATEGORY SUMMARY
# ============================================================

print("\n========== CATEGORY SUMMARY ==========\n")

summary_df = (
    results_df
    .groupby(["Category", "Window", "Top Contrast Percent"])
    .agg(
        Images=("Image", "nunique"),
        Mean_Pixel_AUC=("Pixel ROC-AUC", "mean"),
        Mean_Defect_Coverage=("Defect Coverage", "mean"),
        Mean_Highlight_Precision=("Highlight Precision", "mean"),
        Mean_Contrast_Ratio=("Contrast Ratio", "mean"),
        Mean_Defect_Pixel_Percent=("Defect Pixel Percent", "mean"),
    )
    .reset_index()
    .round(4)
)

print(summary_df.to_string(index=False))

summary_path = (
    OUTPUT_DIR / "defect_mask_contrast_category_summary.csv"
)

summary_df.to_csv(summary_path, index=False)


# ============================================================
# WINDOW COMPARISON
# ============================================================

print("\n========== WINDOW SIZE COMPARISON ==========\n")

window_summary = (
    results_df
    .groupby(["Window", "Top Contrast Percent"])
    .agg(
        Mean_Pixel_AUC=("Pixel ROC-AUC", "mean"),
        Mean_Defect_Coverage=("Defect Coverage", "mean"),
        Mean_Highlight_Precision=("Highlight Precision", "mean"),
        Mean_Contrast_Ratio=("Contrast Ratio", "mean"),
    )
    .reset_index()
    .round(4)
)

print(window_summary.to_string(index=False))

window_path = (
    OUTPUT_DIR / "defect_mask_contrast_window_comparison.csv"
)

window_summary.to_csv(window_path, index=False)


# ============================================================
# PAIR VALIDATION
# ============================================================

pair_df = pd.DataFrame(pair_summary)

print("\n========== IMAGE-MASK PAIR VALIDATION ==========\n")
print(pair_df.to_string(index=False))

pair_path = (
    OUTPUT_DIR / "defect_mask_pair_validation.csv"
)

pair_df.to_csv(pair_path, index=False)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========== FILES SAVED ==========\n")
print(results_path)
print(summary_path)
print(window_path)
print(pair_path)

print("\nEXPERIMENT COMPLETE")
print("No production model or saved model artifacts were changed.")
