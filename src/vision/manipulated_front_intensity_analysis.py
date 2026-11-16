
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data/raw/images/screw/test/manipulated_front"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth/manipulated_front"

OUTPUT_DIR = (
    ROOT
    / "outputs/reports/vision/foreground_method_comparison/intensity_analysis"
)

IMAGE_SIZE = (128, 128)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for image_path in sorted(IMAGE_DIR.glob("*.png")):

        mask_path = MASK_DIR / f"{image_path.stem}_mask.png"

        if not mask_path.exists():
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        image = cv2.resize(
            image, IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask, IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        defect = mask > 0
        normal = ~defect

        if not defect.any() or not normal.any():
            continue

        threshold, _ = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        defect_values = image[defect]
        normal_values = image[normal]

        results.append({
            "Image": image_path.name,
            "Otsu_Threshold": threshold,

            "Defect_Mean_Intensity": defect_values.mean(),
            "Normal_Mean_Intensity": normal_values.mean(),

            "Defect_Median_Intensity": np.median(defect_values),
            "Normal_Median_Intensity": np.median(normal_values),

            "Defect_Darker_Percent": (
                np.mean(defect_values < threshold) * 100
            ),

            "Defect_Brighter_Percent": (
                np.mean(defect_values >= threshold) * 100
            ),

            "Normal_Darker_Percent": (
                np.mean(normal_values < threshold) * 100
            ),

            "Normal_Brighter_Percent": (
                np.mean(normal_values >= threshold) * 100
            ),

            "Defect_Pixel_Count": len(defect_values)
        })

    df = pd.DataFrame(results)

    if df.empty:
        print("No valid results.")
        return

    output_path = OUTPUT_DIR / "manipulated_front_intensity.csv"
    df.to_csv(output_path, index=False)

    print("\n========== IMAGE-WISE INTENSITY ANALYSIS ==========\n")

    print(
        df.round(2).to_string(index=False)
    )

    print("\n========== OVERALL INTENSITY SUMMARY ==========\n")

    print(
        df.select_dtypes(include="number")
        .mean()
        .round(2)
        .to_string()
    )

    print("\nImages analyzed:", len(df))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
