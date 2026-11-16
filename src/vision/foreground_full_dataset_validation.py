
import cv2
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data/raw/images/screw/test"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth"

OUTPUT_DIR = ROOT / "outputs/reports/vision/screw_foreground"

CATEGORIES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]

IMAGE_SIZE = (128, 128)
MORPH_KERNEL_SIZE = 5


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    return mask > 0


# ============================================================
# FOREGROUND SEGMENTATION
# ============================================================

def segment_screw(image):

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:
        raise ValueError("No foreground component detected.")

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    return labels == largest_label


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    print("\n========== FULL DATASET FOREGROUND VALIDATION ==========\n")

    for category in CATEGORIES:

        image_folder = TEST_DIR / category
        mask_folder = MASK_DIR / category

        print(f"\nProcessing: {category}")

        image_paths = sorted([
            p for p in image_folder.iterdir()
            if p.suffix.lower() in {
                ".png", ".jpg", ".jpeg",
                ".bmp", ".tif", ".tiff"
            }
        ])

        processed = 0

        for image_path in image_paths:

            mask_path = mask_folder / f"{image_path.stem}_mask.png"

            if not mask_path.exists():
                errors.append({
                    "Category": category,
                    "Image": image_path.name,
                    "Error": "Ground-truth mask missing"
                })
                continue

            try:
                image = load_image(image_path)
                defect = load_mask(mask_path)
                foreground = segment_screw(image)

                defect_count = np.count_nonzero(defect)
                foreground_count = np.count_nonzero(foreground)

                inside = np.count_nonzero(defect & foreground)
                outside = np.count_nonzero(defect & ~foreground)

                intersection = inside
                union = np.count_nonzero(defect | foreground)

                results.append({
                    "Category": category,
                    "Image": image_path.name,
                    "Foreground_Percent": (
                        100 * foreground_count / foreground.size
                    ),
                    "Defect_Pixel_Percent": (
                        100 * defect_count / defect.size
                    ),
                    "Defect_Inside_Foreground_Percent": (
                        100 * inside / max(defect_count, 1)
                    ),
                    "Defect_Outside_Foreground_Percent": (
                        100 * outside / max(defect_count, 1)
                    ),
                    "Foreground_Defect_Overlap_IoU": (
                        intersection / max(union, 1)
                    )
                })

                processed += 1

            except Exception as exc:
                errors.append({
                    "Category": category,
                    "Image": image_path.name,
                    "Error": str(exc)
                })

        print(f"Successfully processed: {processed}")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    df = pd.DataFrame(results)

    output_csv = OUTPUT_DIR / "foreground_full_dataset_results.csv"
    df.to_csv(output_csv, index=False)

    if not df.empty:

        category_summary = (
            df.groupby("Category")
            .agg(
                Images=("Image", "count"),
                Mean_Foreground_Percent=("Foreground_Percent", "mean"),
                Mean_Defect_Pixel_Percent=("Defect_Pixel_Percent", "mean"),
                Mean_Defect_Inside_Percent=(
                    "Defect_Inside_Foreground_Percent", "mean"
                ),
                Mean_Defect_Outside_Percent=(
                    "Defect_Outside_Foreground_Percent", "mean"
                ),
                Mean_Overlap_IoU=("Foreground_Defect_Overlap_IoU", "mean")
            )
            .reset_index()
            .round(4)
        )

        summary_path = (
            OUTPUT_DIR / "foreground_full_dataset_summary.csv"
        )

        category_summary.to_csv(summary_path, index=False)

        print("\n========== CATEGORY SUMMARY ==========")
        print(category_summary.to_string(index=False))

        print("\nTotal images processed:", len(df))

    if errors:
        error_df = pd.DataFrame(errors)
        error_path = OUTPUT_DIR / "foreground_validation_errors.csv"
        error_df.to_csv(error_path, index=False)

        print("\nErrors or missing masks:", len(errors))
        print(f"Error report: {error_path}")

    print(f"\nDetailed results saved: {output_csv}")
    print("\nEXPERIMENT COMPLETE")
    print("No production model or saved model artifacts were changed.")


if __name__ == "__main__":
    main()
