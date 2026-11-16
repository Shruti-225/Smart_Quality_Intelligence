
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data/raw/images/screw/test/manipulated_front"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth/manipulated_front"

OUTPUT_DIR = (
    ROOT
    / "outputs/reports/vision/foreground_method_comparison/polarity_test"
)

IMAGE_SIZE = (128, 128)
KERNEL_SIZE = 5


def segment(image, polarity):

    threshold_type = (
        cv2.THRESH_BINARY_INV
        if polarity == "dark_foreground"
        else cv2.THRESH_BINARY
    )

    threshold, binary = cv2.threshold(
        image,
        0,
        255,
        threshold_type + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE)
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
        return None, threshold, 0, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    foreground = labels == largest_label

    largest_share = (
        areas.max() / max(areas.sum(), 1) * 100
    )

    return foreground, threshold, count - 1, largest_share


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for image_path in sorted(TEST_DIR.glob("*.png")):

        mask_path = (
            MASK_DIR / f"{image_path.stem}_mask.png"
        )

        if not mask_path.exists():
            print(f"Missing mask: {image_path.name}")
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

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
        defect_count = int(defect.sum())

        if defect_count == 0:
            continue

        defect_mean = float(image[defect].mean())
        normal_mean = float(image[~defect].mean())

        for polarity in [
            "dark_foreground",
            "bright_foreground"
        ]:

            foreground, threshold, components, share = segment(
                image, polarity
            )

            if foreground is None:
                continue

            foreground_count = int(foreground.sum())
            inside = int((defect & foreground).sum())

            results.append({
                "Image": image_path.name,
                "Polarity": polarity,
                "Otsu_Threshold": threshold,
                "Foreground_Percent": (
                    foreground_count / foreground.size * 100
                ),
                "Defect_Inside_Percent": (
                    inside / defect_count * 100
                ),
                "Defect_Outside_Percent": (
                    100 - inside / defect_count * 100
                ),
                "Connected_Components": components,
                "Largest_Component_Share_Percent": share,
                "Defect_Mean_Intensity": defect_mean,
                "Normal_Mean_Intensity": normal_mean
            })

    if not results:
        print("No valid results.")
        return

    df = pd.DataFrame(results)

    detailed_path = OUTPUT_DIR / "polarity_detailed.csv"
    summary_path = OUTPUT_DIR / "polarity_summary.csv"

    df.to_csv(detailed_path, index=False)

    summary = (
        df.groupby("Polarity")
        .mean(numeric_only=True)
        .round(4)
        .reset_index()
    )

    summary.to_csv(summary_path, index=False)

    print("\n========== POLARITY SUMMARY ==========\n")
    print(summary.to_string(index=False))

    print("\n========== PER-IMAGE RESULTS ==========\n")
    print(
        df.pivot(
            index="Image",
            columns="Polarity",
            values="Defect_Inside_Percent"
        ).round(2).to_string()
    )

    print("\nImages evaluated:", df["Image"].nunique())
    print("Total evaluations:", len(df))
    print("Detailed report:", detailed_path)
    print("Summary report:", summary_path)

    print("\nProduction model and artifacts unchanged.")


if __name__ == "__main__":
    main()
