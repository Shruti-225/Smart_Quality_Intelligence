
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data" / "raw" / "images" / "screw" / "test"
MASK_DIR = ROOT / "data" / "raw" / "images" / "screw" / "ground_truth"

OUTPUT_DIR = ROOT / "outputs" / "reports" / "vision" / "contrast_overlays"

CATEGORIES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]

IMAGE_SIZE = (128, 128)
WINDOW_SIZE = 9
SAMPLES_PER_CATEGORY = 2
MASK_THRESHOLD = 0


# ============================================================
# IMAGE PROCESSING
# ============================================================

def load_image(path):

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

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

    return image.astype(np.float32)


def load_mask(path):

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not read mask: {path}")

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    return mask > MASK_THRESHOLD


def calculate_local_contrast(image):

    mean = cv2.boxFilter(
        image,
        ddepth=-1,
        ksize=(WINDOW_SIZE, WINDOW_SIZE),
        normalize=True
    )

    mean_squared = cv2.boxFilter(
        image ** 2,
        ddepth=-1,
        ksize=(WINDOW_SIZE, WINDOW_SIZE),
        normalize=True
    )

    variance = np.maximum(mean_squared - mean ** 2, 0)

    return np.sqrt(variance)


def create_mask_overlay(image, mask):

    normalized = image / 255.0

    rgb = np.stack(
        [normalized, normalized, normalized],
        axis=-1
    )

    # Blend defect pixels with red
    rgb[mask] = (
        0.45 * rgb[mask] +
        0.55 * np.array([1.0, 0.0, 0.0])
    )

    return np.clip(rgb, 0, 1)


# ============================================================
# VISUALIZATION
# ============================================================

def create_category_visualization(category, samples):

    if not samples:
        print(f"No valid samples for {category}")
        return

    fig, axes = plt.subplots(
        len(samples),
        4,
        figsize=(14, 4.2 * len(samples))
    )

    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        f"{category} - 9x9 Local Contrast Analysis",
        fontsize=15,
        fontweight="bold"
    )

    for row, sample in enumerate(samples):

        image = sample["image"]
        mask = sample["mask"]
        contrast = sample["contrast"]
        image_name = sample["name"]

        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Original: {image_name}")

        axes[row, 1].imshow(create_mask_overlay(image, mask))
        axes[row, 1].set_title("Ground-truth mask overlay")

        axes[row, 2].imshow(contrast, cmap="inferno")
        axes[row, 2].set_title("9x9 local contrast")

        axes[row, 3].imshow(contrast, cmap="inferno")
        axes[row, 3].contour(
            mask.astype(float),
            levels=[0.5],
            colors=["cyan"],
            linewidths=1
        )
        axes[row, 3].set_title("Contrast + defect boundary")

        for col in range(4):
            axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = OUTPUT_DIR / f"{category}_overlay.png"

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def create_combined_overview(category_samples):

    fig, axes = plt.subplots(
        len(CATEGORIES),
        4,
        figsize=(14, 17)
    )

    fig.suptitle(
        "Smart Quality - Defect Mask and Local Contrast Overview",
        fontsize=16,
        fontweight="bold"
    )

    for row, category in enumerate(CATEGORIES):

        if not category_samples[category]:
            continue

        sample = category_samples[category][0]

        image = sample["image"]
        mask = sample["mask"]
        contrast = sample["contrast"]

        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_ylabel(
            category,
            fontsize=9,
            fontweight="bold"
        )

        axes[row, 1].imshow(create_mask_overlay(image, mask))

        axes[row, 2].imshow(contrast, cmap="inferno")

        axes[row, 3].imshow(contrast, cmap="inferno")
        axes[row, 3].contour(
            mask.astype(float),
            levels=[0.5],
            colors=["cyan"],
            linewidths=1
        )

        for col in range(4):
            axes[row, col].axis("off")

    for col, title in enumerate([
        "Original",
        "Ground-truth mask",
        "9x9 contrast",
        "Contrast + mask boundary"
    ]):
        axes[0, col].set_title(title, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUT_DIR / "all_categories_overview.png"

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    category_samples = {}

    print("\n========== DEFECT CONTRAST VISUALIZATION ==========\n")

    for category in CATEGORIES:

        print(f"\nProcessing: {category}")

        image_folder = TEST_DIR / category
        mask_folder = MASK_DIR / category

        if not image_folder.exists() or not mask_folder.exists():
            print("Missing image or mask folder. Skipping.")
            category_samples[category] = []
            continue

        image_paths = sorted([
            p for p in image_folder.iterdir()
            if p.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
            }
        ])

        samples = []

        for image_path in image_paths:

            mask_path = mask_folder / f"{image_path.stem}_mask.png"

            if not mask_path.exists():
                print(f"Missing mask: {mask_path.name}")
                continue

            image = load_image(image_path)
            mask = load_mask(mask_path)
            contrast = calculate_local_contrast(image)

            samples.append({
                "name": image_path.name,
                "image": image,
                "mask": mask,
                "contrast": contrast
            })

            if len(samples) >= SAMPLES_PER_CATEGORY:
                break

        category_samples[category] = samples

        print(f"Selected images: {len(samples)}")

        create_category_visualization(category, samples)

    create_combined_overview(category_samples)

    print("\n========== EXPERIMENT COMPLETE ==========")
    print(f"Output folder: {OUTPUT_DIR}")
    print("No production model or saved model artifacts were changed.")


if __name__ == "__main__":
    main()
