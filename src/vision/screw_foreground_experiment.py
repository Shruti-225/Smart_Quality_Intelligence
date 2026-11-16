import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data/raw/images/screw/test"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth"

OUTPUT_DIR = (
    ROOT / "outputs/reports/vision/screw_foreground"
)

CATEGORIES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]

IMAGE_SIZE = (128, 128)
SAMPLES_PER_CATEGORY = 2
MORPH_KERNEL_SIZE = 5


# ============================================================
# LOAD IMAGE AND GROUND-TRUTH MASK
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


def load_defect_mask(path):

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
# SCREW FOREGROUND SEGMENTATION
# ============================================================

def segment_screw(image):

    # Otsu thresholding: screw is expected to be darker
    # than its surrounding background.
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

    # Remove small noise and close small gaps.
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

    # Find connected components.
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:
        raise ValueError("No foreground component detected.")

    # Ignore background label 0.
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))

    foreground = labels == largest_label

    return foreground


# ============================================================
# VISUALIZATION
# ============================================================

def create_overlay(image, mask, color):

    normalized = image.astype(np.float32) / 255.0

    rgb = np.stack(
        [normalized, normalized, normalized],
        axis=-1
    )

    rgb[mask] = (
        0.45 * rgb[mask] +
        0.55 * np.array(color)
    )

    return np.clip(rgb, 0, 1)


def visualize_category(category, samples):

    if not samples:
        return

    fig, axes = plt.subplots(
        len(samples),
        4,
        figsize=(14, 4.5 * len(samples))
    )

    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        f"{category} - Screw Foreground Segmentation",
        fontsize=15,
        fontweight="bold"
    )

    for row, sample in enumerate(samples):

        image = sample["image"]
        foreground = sample["foreground"]
        defect = sample["defect"]
        name = sample["name"]

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"Original: {name}")

        axes[row, 1].imshow(
            create_overlay(
                image,
                foreground,
                [0, 1, 0]
            )
        )
        axes[row, 1].set_title("Screw foreground (green)")

        axes[row, 2].imshow(
            create_overlay(
                image,
                defect,
                [1, 0, 0]
            )
        )
        axes[row, 2].set_title("Ground-truth defect (red)")

        axes[row, 3].imshow(
            create_overlay(
                image,
                foreground,
                [0, 1, 0]
            )
        )

        axes[row, 3].contour(
            defect.astype(float),
            levels=[0.5],
            colors=["red"],
            linewidths=1
        )

        axes[row, 3].set_title(
            "Foreground + defect boundary"
        )

        for col in range(4):
            axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output = OUTPUT_DIR / f"{category}_segmentation.png"

    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []
    summary = []

    print("\n========== SCREW FOREGROUND EXPERIMENT ==========\n")

    for category in CATEGORIES:

        print(f"\nProcessing: {category}")

        image_folder = TEST_DIR / category
        mask_folder = MASK_DIR / category

        samples = []

        image_paths = sorted([
            p for p in image_folder.iterdir()
            if p.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
            }
        ])

        for image_path in image_paths:

            mask_path = mask_folder / (
                f"{image_path.stem}_mask.png"
            )

            if not mask_path.exists():
                continue

            image = load_image(image_path)
            defect = load_defect_mask(mask_path)

            foreground = segment_screw(image)

            defect_pixels = np.count_nonzero(defect)
            foreground_pixels = np.count_nonzero(foreground)

            defect_inside = np.count_nonzero(
                defect & foreground
            )

            defect_outside = np.count_nonzero(
                defect & ~foreground
            )

            summary.append({
                "Category": category,
                "Image": image_path.name,
                "Foreground_Percent": (
                    100 * foreground_pixels / foreground.size
                ),
                "Defect_Inside_Foreground_Percent": (
                    100 * defect_inside / max(defect_pixels, 1)
                ),
                "Defect_Outside_Foreground_Percent": (
                    100 * defect_outside / max(defect_pixels, 1)
                )
            })

            samples.append({
                "name": image_path.name,
                "image": image,
                "foreground": foreground,
                "defect": defect
            })

            if len(samples) >= SAMPLES_PER_CATEGORY:
                break

        print(f"Selected images: {len(samples)}")

        visualize_category(category, samples)

        all_samples.extend(samples)

    # Save numerical validation results.
    import pandas as pd

    df = pd.DataFrame(summary)

    csv_path = OUTPUT_DIR / "foreground_segmentation_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n========== SEGMENTATION SUMMARY ==========")

    if not df.empty:
        print(
            df.groupby("Category")[
                [
                    "Foreground_Percent",
                    "Defect_Inside_Foreground_Percent",
                    "Defect_Outside_Foreground_Percent"
                ]
            ].mean().round(3).to_string()
        )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Output folder: {OUTPUT_DIR}")

    print("\nEXPERIMENT COMPLETE")
    print("No production model or saved model artifacts were changed.")


if __name__ == "__main__":
    main()
