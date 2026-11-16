
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data/raw/images/screw/test"
MASK_DIR = ROOT / "data/raw/images/screw/ground_truth"

OUTPUT_DIR = (
    ROOT
    / "outputs/reports/vision/foreground_method_comparison/visual_diagnostics"
)

IMAGE_SIZE = (128, 128)
MORPH_KERNEL_SIZE = 5

SELECTED_IMAGES = {
    "manipulated_front": ["000.png", "002.png", "017.png", "020.png", "021.png"],
    "scratch_neck": ["003.png", "013.png", "016.png", "019.png"],
    "thread_top": ["014.png"],
}


# ============================================================
# OTSU WITHOUT MORPHOLOGICAL OPENING
# ============================================================

def segment_screw_no_opening(image):

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

    # Keep closing, but intentionally skip opening.
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

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))

    foreground = labels == largest_label

    return foreground


# ============================================================
# LOAD IMAGE AND MASK
# ============================================================

def load_sample(category, filename):

    image_path = TEST_DIR / category / filename

    mask_filename = Path(filename).stem + "_mask.png"
    mask_path = MASK_DIR / category / mask_filename

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    defect_mask = mask > 0
    foreground = segment_screw_no_opening(image)

    return image, defect_mask, foreground


# ============================================================
# CREATE COMPARISON VISUALIZATION
# ============================================================

def create_visualization():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = []

    for category, filenames in SELECTED_IMAGES.items():

        for filename in filenames:

            print(f"Processing: {category}/{filename}")

            image, defect_mask, foreground = load_sample(
                category,
                filename
            )

            samples.append(
                (category, filename, image, defect_mask, foreground)
            )

    fig, axes = plt.subplots(
        len(samples),
        4,
        figsize=(16, 4 * len(samples))
    )

    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Original Image",
        "Ground Truth Defect Mask",
        "Otsu No Opening Foreground",
        "Defect-Foreground Comparison"
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, sample in enumerate(samples):

        category, filename, image, defect_mask, foreground = sample

        # Panel 1: Original image
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_ylabel(
            f"{category}\n{filename}",
            fontsize=9
        )

        # Panel 2: Ground-truth defect mask
        axes[row, 1].imshow(defect_mask, cmap="gray")

        # Panel 3: Predicted foreground
        axes[row, 2].imshow(foreground, cmap="gray")

        # Panel 4: Green = defect inside foreground
        #          Red = defect outside foreground
        comparison = np.zeros(
            (*defect_mask.shape, 3),
            dtype=np.uint8
        )

        inside = defect_mask & foreground
        outside = defect_mask & ~foreground

        comparison[inside] = [0, 255, 0]
        comparison[outside] = [255, 0, 0]

        axes[row, 3].imshow(comparison)

        for col in range(4):
            axes[row, col].axis("off")

    fig.suptitle(
        "Otsu Without Opening: Defect and Foreground Comparison",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = OUTPUT_DIR / "no_opening_visual_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\n========================================")
    print("VISUALIZATION COMPLETED")
    print("========================================")
    print(f"Saved: {output_path}")
    print(f"Total images: {len(samples)}")


if __name__ == "__main__":
    create_visualization()
