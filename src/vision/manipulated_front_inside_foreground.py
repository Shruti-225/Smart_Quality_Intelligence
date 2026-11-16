
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data/raw/images/screw/test/manipulated_front"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth/manipulated_front"

OUTPUT_DIR = (
    ROOT / "outputs/reports/vision/foreground_method_comparison"
    / "inside_foreground_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
LOCAL_WINDOW_SIZE = 9


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    return mask > 0


def segment_screw(image):
    # Dark-screw Otsu threshold, closing, and largest component.
    # No morphological opening.
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if count <= 1:
        return binary > 0

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest_label


def local_std(image):
    image = image.astype(np.float32)
    mean = cv2.blur(image, (LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE))
    mean_sq = cv2.blur(image ** 2, (LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE))
    variance = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(variance)


def mean(values):
    return float(np.mean(values)) if values.size else np.nan


def median(values):
    return float(np.median(values)) if values.size else np.nan


records = []

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    mask_path = MASK_DIR / f"{image_path.stem}_mask.png"

    if not mask_path.exists():
        print(f"Missing mask: {mask_path.name}")
        continue

    image = load_image(image_path)
    defect_mask = load_mask(mask_path)
    foreground = segment_screw(image)

    # Compare defect and non-defect pixels only inside foreground.
    defect_inside = defect_mask & foreground
    normal_inside = (~defect_mask) & foreground

    contrast = local_std(image)

    defect_intensity = image[defect_inside]
    normal_intensity = image[normal_inside]

    defect_contrast = contrast[defect_inside]
    normal_contrast = contrast[normal_inside]

    total_defect = np.count_nonzero(defect_mask)
    retained_defect = np.count_nonzero(defect_inside)

    retention = (
        100 * retained_defect / total_defect if total_defect else np.nan
    )

    defect_contrast_mean = mean(defect_contrast)
    normal_contrast_mean = mean(normal_contrast)

    record = {
        "Image": image_path.name,
        "Foreground_Percent": 100 * np.mean(foreground),
        "Defect_Retention_Percent": retention,
        "Defect_Pixels_Inside": retained_defect,
        "Normal_Pixels_Inside": np.count_nonzero(normal_inside),
        "Defect_Mean_Intensity": mean(defect_intensity),
        "Normal_Mean_Intensity": mean(normal_intensity),
        "Defect_Median_Intensity": median(defect_intensity),
        "Normal_Median_Intensity": median(normal_intensity),
        "Defect_Mean_Local_Contrast": defect_contrast_mean,
        "Normal_Mean_Local_Contrast": normal_contrast_mean,
        "Defect_Median_Local_Contrast": median(defect_contrast),
        "Normal_Median_Local_Contrast": median(normal_contrast),
        "Local_Contrast_Mean_Difference": (
            defect_contrast_mean - normal_contrast_mean
        ),
        "Local_Contrast_Mean_Ratio": (
            defect_contrast_mean / normal_contrast_mean
            if np.isfinite(normal_contrast_mean) and normal_contrast_mean != 0
            else np.nan
        ),
    }

    records.append(record)

    print(
        f"{image_path.name}: "
        f"retention={retention:.2f}%, "
        f"defect contrast={defect_contrast_mean:.2f}, "
        f"normal contrast={normal_contrast_mean:.2f}"
    )

if not records:
    raise RuntimeError("No image-mask pairs were processed.")

df = pd.DataFrame(records)

detailed_path = OUTPUT_DIR / "manipulated_front_inside_foreground_detailed.csv"
summary_path = OUTPUT_DIR / "manipulated_front_inside_foreground_summary.csv"

df.to_csv(detailed_path, index=False)

numeric_columns = df.select_dtypes(include=[np.number]).columns
summary = df[numeric_columns].mean().to_frame("Mean").T
summary.insert(0, "Images_Analyzed", len(df))
summary.to_csv(summary_path, index=False)

print("\n========== OVERALL INSIDE-FOREGROUND SUMMARY ==========\n")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n========== IMAGE-WISE CONTRAST COMPARISON ==========\n")
print(
    df[
        [
            "Image",
            "Defect_Retention_Percent",
            "Defect_Mean_Local_Contrast",
            "Normal_Mean_Local_Contrast",
            "Local_Contrast_Mean_Difference",
            "Local_Contrast_Mean_Ratio",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
)

print(f"\nImages analyzed: {len(df)}")
print(f"Detailed CSV: {detailed_path}")
print(f"Summary CSV: {summary_path}")
