
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data/raw/images/screw/test"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth"

OUTPUT_DIR = (
    ROOT / "outputs/reports/vision/"
    "foreground_method_comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)

CATEGORIES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top"
]

# ==========================================
# SEGMENTATION CONFIGURATIONS
# ==========================================

METHODS = {
    "otsu_3x3": {
        "kernel": 3,
        "opening": True
    },
    "otsu_5x5_baseline": {
        "kernel": 5,
        "opening": True
    },
    "otsu_7x7": {
        "kernel": 7,
        "opening": True
    },
    "otsu_no_opening": {
        "kernel": 5,
        "opening": False
    }
}


# ==========================================
# IMAGE LOADING
# ==========================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )


def load_mask(path):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    return mask > 0


# ==========================================
# SEGMENTATION
# ==========================================

def segment_screw(image, kernel_size, use_opening):

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    if use_opening:

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

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    foreground = np.zeros_like(binary)

    if count > 1:

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        foreground[labels == largest_label] = 255

    return foreground > 0


# ==========================================
# EVALUATION
# ==========================================

def evaluate_method(image, defect_mask, config):

    foreground = segment_screw(
        image,
        config["kernel"],
        config["opening"]
    )

    total_pixels = defect_mask.sum()
    total_image_pixels = defect_mask.size

    inside_pixels = np.sum(
        defect_mask & foreground
    )

    outside_pixels = np.sum(
        defect_mask & (~foreground)
    )

    defect_retention = (
        inside_pixels / total_pixels * 100
        if total_pixels else 0
    )

    defect_outside = (
        outside_pixels / total_pixels * 100
        if total_pixels else 0
    )

    foreground_percent = (
        foreground.sum() / total_image_pixels * 100
    )

    return {
        "Foreground_Percent": foreground_percent,
        "Defect_Inside_Percent": defect_retention,
        "Defect_Outside_Percent": defect_outside,
        "Defect_Pixels": int(total_pixels),
        "Foreground_Pixels": int(foreground.sum())
    }


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    print("\n========== FOREGROUND METHOD COMPARISON ==========\n")

    results = []
    errors = []

    total_processed = 0

    for category in CATEGORIES:

        image_folder = IMAGE_DIR / category
        mask_folder = MASK_DIR / category

        image_paths = sorted(image_folder.glob("*.png"))

        if not image_paths:
            image_paths = sorted(image_folder.glob("*.jpg"))

        print(f"\nProcessing: {category}")

        category_count = 0

        for image_path in image_paths:

            mask_path = (
                mask_folder /
                f"{image_path.stem}_mask.png"
            )

            if not mask_path.exists():

                errors.append({
                    "Category": category,
                    "Image": image_path.name,
                    "Error": "Missing ground-truth mask"
                })
                continue

            try:

                image = load_image(image_path)
                defect_mask = load_mask(mask_path)

                for method_name, config in METHODS.items():

                    metrics = evaluate_method(
                        image,
                        defect_mask,
                        config
                    )

                    results.append({
                        "Category": category,
                        "Image": image_path.name,
                        "Method": method_name,
                        **metrics
                    })

                category_count += 1
                total_processed += 1

            except Exception as error:

                errors.append({
                    "Category": category,
                    "Image": image_path.name,
                    "Error": str(error)
                })

        print("Successfully processed:", category_count)

    # ==========================================
    # SAVE DETAILED RESULTS
    # ==========================================

    results_df = pd.DataFrame(results)

    detailed_path = (
        OUTPUT_DIR / "foreground_method_detailed_results.csv"
    )

    results_df.to_csv(
        detailed_path,
        index=False
    )

    # ==========================================
    # CATEGORY SUMMARY
    # ==========================================

    summary = (
        results_df.groupby(
            ["Category", "Method"]
        )
        .agg(
            Images=("Image", "count"),
            Mean_Foreground_Percent=(
                "Foreground_Percent", "mean"
            ),
            Mean_Defect_Inside_Percent=(
                "Defect_Inside_Percent", "mean"
            ),
            Mean_Defect_Outside_Percent=(
                "Defect_Outside_Percent", "mean"
            )
        )
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR / "foreground_method_category_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ==========================================
    # OVERALL SUMMARY
    # ==========================================

    overall = (
        results_df.groupby("Method")
        .agg(
            Images=("Image", "count"),
            Mean_Foreground_Percent=(
                "Foreground_Percent", "mean"
            ),
            Mean_Defect_Inside_Percent=(
                "Defect_Inside_Percent", "mean"
            ),
            Mean_Defect_Outside_Percent=(
                "Defect_Outside_Percent", "mean"
            ),
            Std_Defect_Inside_Percent=(
                "Defect_Inside_Percent", "std"
            )
        )
        .reset_index()
    )

    overall_path = (
        OUTPUT_DIR / "foreground_method_overall_summary.csv"
    )

    overall.to_csv(
        overall_path,
        index=False
    )

    # ==========================================
    # SAVE ERRORS
    # ==========================================

    if errors:

        errors_df = pd.DataFrame(errors)

        errors_path = OUTPUT_DIR / "foreground_method_errors.csv"

        errors_df.to_csv(
            errors_path,
            index=False
        )

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    print("\n========== OVERALL METHOD SUMMARY ==========\n")

    if not overall.empty:

        print(
            overall.round(4).to_string(index=False)
        )

    print("\n========== CATEGORY SUMMARY ==========\n")

    if not summary.empty:

        print(
            summary.round(4).to_string(index=False)
        )

    print("\nTotal images processed:", total_processed)
    print("Total method evaluations:", len(results))

    print("\nDetailed results:", detailed_path)
    print("Category summary:", summary_path)
    print("Overall summary:", overall_path)

    if errors:
        print("\nErrors encountered:", len(errors))
        print("Error report saved.")

    print("\n========== EXPERIMENT COMPLETE ==========")
    print("Production model and saved artifacts were not changed.")


if __name__ == "__main__":
    main()
