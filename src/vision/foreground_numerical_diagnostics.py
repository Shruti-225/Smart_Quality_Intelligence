
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data/raw/images/screw/test"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth"

OUTPUT_DIR = (
    ROOT
    / "outputs/reports/vision/foreground_method_comparison/numerical_diagnostics"
)

IMAGE_SIZE = (128, 128)
KERNEL_SIZE = 5


# ============================================================
# FOREGROUND SEGMENTATION
# ============================================================

def segment_foreground(image, method):

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    if method == "otsu_baseline":
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel
        )

    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:
        return None, 0, 0

    component_areas = stats[1:, cv2.CC_STAT_AREA]

    largest_label = 1 + int(np.argmax(component_areas))
    largest_area = int(component_areas.max())
    total_component_area = int(component_areas.sum())

    foreground = labels == largest_label

    return foreground, count - 1, largest_area / max(
        total_component_area, 1
    ) * 100


# ============================================================
# IMAGE DIAGNOSTICS
# ============================================================

def analyze_image(category, image_path, method):

    mask_path = (
        MASK_DIR
        / category
        / f"{image_path.stem}_mask.png"
    )

    if not mask_path.exists():
        print(f"Missing mask: {mask_path}")
        return None

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        print(f"Could not read: {image_path}")
        return None

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    defect = mask > 0

    foreground, component_count, largest_share = (
        segment_foreground(image, method)
    )

    if foreground is None:
        return None

    total_pixels = foreground.size
    foreground_pixels = int(foreground.sum())
    defect_pixels = int(defect.sum())

    if foreground_pixels == 0 or defect_pixels == 0:
        return None

    defect_inside = int((defect & foreground).sum())
    defect_outside = int((defect & ~foreground).sum())

    # Foreground touching image border
    border = np.zeros_like(foreground, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    border_pixels = int((foreground & border).sum())

    # Bounding box occupancy
    ys, xs = np.where(foreground)

    bbox_area = (
        (ys.max() - ys.min() + 1)
        * (xs.max() - xs.min() + 1)
    )

    bbox_fill = foreground_pixels / max(bbox_area, 1) * 100

    # Intensity statistics
    foreground_intensity = image[foreground]
    outside_intensity = image[~foreground]

    return {
        "Category": category,
        "Image": image_path.name,
        "Method": method,

        "Foreground_Percent": (
            foreground_pixels / total_pixels * 100
        ),

        "Defect_Inside_Percent": (
            defect_inside / defect_pixels * 100
        ),

        "Defect_Outside_Percent": (
            defect_outside / defect_pixels * 100
        ),

        "Foreground_Border_Percent": (
            border_pixels / foreground_pixels * 100
        ),

        "Largest_Component_Share_Percent": largest_share,

        "Connected_Component_Count": component_count,

        "Bounding_Box_Fill_Percent": bbox_fill,

        "Foreground_Mean_Intensity": (
            float(np.mean(foreground_intensity))
        ),

        "Outside_Mean_Intensity": (
            float(np.mean(outside_intensity))
            if len(outside_intensity) else np.nan
        ),

        "Foreground_Intensity_Std": (
            float(np.std(foreground_intensity))
        ),

        "Defect_Pixels": defect_pixels,
        "Foreground_Pixels": foreground_pixels
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = [
        "otsu_baseline",
        "otsu_no_opening"
    ]

    results = []

    categories = sorted(
        path.name for path in TEST_DIR.iterdir()
        if path.is_dir()
    )

    for category in categories:

        category_dir = TEST_DIR / category

        image_paths = sorted(category_dir.glob("*.png"))

        print(f"\nProcessing: {category}")

        for image_path in image_paths:

            for method in methods:

                result = analyze_image(
                    category,
                    image_path,
                    method
                )

                if result is not None:
                    results.append(result)

        print(f"Images found: {len(image_paths)}")

    if not results:
        print("No valid results generated.")
        return

    df = pd.DataFrame(results)

    detailed_path = OUTPUT_DIR / "foreground_diagnostics_detailed.csv"

    df.to_csv(detailed_path, index=False)

    numeric_columns = [
        "Foreground_Percent",
        "Defect_Inside_Percent",
        "Defect_Outside_Percent",
        "Foreground_Border_Percent",
        "Largest_Component_Share_Percent",
        "Connected_Component_Count",
        "Bounding_Box_Fill_Percent",
        "Foreground_Mean_Intensity",
        "Outside_Mean_Intensity",
        "Foreground_Intensity_Std"
    ]

    overall = (
        df.groupby("Method")[numeric_columns]
        .mean()
        .round(4)
        .reset_index()
    )

    category_summary = (
        df.groupby(["Category", "Method"])[numeric_columns]
        .mean()
        .round(4)
        .reset_index()
    )

    overall_path = OUTPUT_DIR / "foreground_diagnostics_overall.csv"
    category_path = OUTPUT_DIR / "foreground_diagnostics_category.csv"

    overall.to_csv(overall_path, index=False)
    category_summary.to_csv(category_path, index=False)

    print("\n========== OVERALL DIAGNOSTICS ==========\n")
    print(overall.to_string(index=False))

    print("\n========== CATEGORY DIAGNOSTICS ==========\n")
    print(category_summary.to_string(index=False))

    print("\n========== EXPERIMENT SUMMARY ==========")
    print(f"Total evaluations: {len(df)}")
    print(f"Unique images: {df['Image'].nunique()}")

    print(f"\nDetailed: {detailed_path}")
    print(f"Overall: {overall_path}")
    print(f"Category: {category_path}")

    print("\nProduction model and saved artifacts were not changed.")


if __name__ == "__main__":
    main()
