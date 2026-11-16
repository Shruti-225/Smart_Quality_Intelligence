
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data/raw/images/screw/test/manipulated_front"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth/manipulated_front"

OUTPUT_DIR = (
    ROOT / "outputs/reports/vision/foreground_method_comparison"
    / "pixel_auc_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
WINDOW_SIZE = 9
TOP_PERCENT = 10


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
    mean = cv2.blur(image, (WINDOW_SIZE, WINDOW_SIZE))
    mean_sq = cv2.blur(image ** 2, (WINDOW_SIZE, WINDOW_SIZE))
    variance = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(variance)


records = []

all_labels = []
all_contrast_scores = []
all_intensity_scores = []

print("\n========== PER-IMAGE PIXEL ROC-AUC ==========\n")

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    mask_path = MASK_DIR / f"{image_path.stem}_mask.png"

    if not mask_path.exists():
        print(f"Skipping missing mask: {mask_path.name}")
        continue

    image = load_image(image_path)
    defect_mask = load_mask(mask_path)
    foreground = segment_screw(image)

    # Evaluate only pixels retained inside the predicted screw foreground.
    labels = defect_mask[foreground].astype(np.uint8)
    contrast_scores = local_std(image)[foreground]
    intensity_scores = image[foreground].astype(np.float32)

    if len(np.unique(labels)) < 2:
        print(f"Skipping {image_path.name}: only one class inside foreground")
        continue

    contrast_auc = roc_auc_score(labels, contrast_scores)
    intensity_auc = roc_auc_score(labels, intensity_scores)

    # Top 10% highest local-contrast pixels inside foreground.
    top_count = max(1, int(np.ceil(len(contrast_scores) * TOP_PERCENT / 100)))
    top_indices = np.argsort(contrast_scores)[-top_count:]

    selected_labels = labels[top_indices]
    true_positive = int(np.sum(selected_labels == 1))
    total_defects_inside = int(np.sum(labels == 1))

    precision = true_positive / top_count
    recall = (
        true_positive / total_defects_inside
        if total_defects_inside else np.nan
    )

    records.append({
        "Image": image_path.name,
        "Foreground_Pixel_Count": len(labels),
        "Defect_Pixels_Inside": total_defects_inside,
        "Contrast_ROC_AUC": contrast_auc,
        "Intensity_ROC_AUC": intensity_auc,
        "Top10_Contrast_Precision": precision,
        "Top10_Contrast_Recall": recall,
    })

    all_labels.extend(labels.tolist())
    all_contrast_scores.extend(contrast_scores.tolist())
    all_intensity_scores.extend(intensity_scores.tolist())

    print(
        f"{image_path.name}: "
        f"contrast AUC={contrast_auc:.4f}, "
        f"intensity AUC={intensity_auc:.4f}, "
        f"top10 precision={precision:.4f}, "
        f"top10 recall={recall:.4f}"
    )

if not records:
    raise RuntimeError("No valid images were evaluated.")

df = pd.DataFrame(records)

# Macro average: each image contributes equally.
macro_contrast_auc = df["Contrast_ROC_AUC"].mean()
macro_intensity_auc = df["Intensity_ROC_AUC"].mean()

macro_precision = df["Top10_Contrast_Precision"].mean()
macro_recall = df["Top10_Contrast_Recall"].mean()

# Pooled AUC: all retained pixels are evaluated together.
pooled_labels = np.asarray(all_labels)
pooled_contrast = np.asarray(all_contrast_scores)
pooled_intensity = np.asarray(all_intensity_scores)

pooled_contrast_auc = roc_auc_score(pooled_labels, pooled_contrast)
pooled_intensity_auc = roc_auc_score(pooled_labels, pooled_intensity)

summary = pd.DataFrame([{
    "Images_Analyzed": len(df),
    "Total_Foreground_Pixels": len(pooled_labels),
    "Total_Defect_Pixels_Inside": int(np.sum(pooled_labels)),
    "Macro_Contrast_ROC_AUC": macro_contrast_auc,
    "Pooled_Contrast_ROC_AUC": pooled_contrast_auc,
    "Macro_Intensity_ROC_AUC": macro_intensity_auc,
    "Pooled_Intensity_ROC_AUC": pooled_intensity_auc,
    "Macro_Top10_Contrast_Precision": macro_precision,
    "Macro_Top10_Contrast_Recall": macro_recall,
}])

detailed_path = OUTPUT_DIR / "manipulated_front_pixel_auc_detailed.csv"
summary_path = OUTPUT_DIR / "manipulated_front_pixel_auc_summary.csv"

df.to_csv(detailed_path, index=False)
summary.to_csv(summary_path, index=False)

print("\n========== OVERALL PIXEL DETECTION SUMMARY ==========\n")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n========== IMAGE-WISE AUC DETAILS ==========\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nImages analyzed: {len(df)}")
print(f"Detailed CSV: {detailed_path}")
print(f"Summary CSV: {summary_path}")
