
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import vision_model as vm
import normalization_experiment as ne


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

NORMALIZATION = "Percentile"
IMAGE_SIZE = 128
GRID_SIZE = 8

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "vision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

def load_normalized_image(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = ne.normalize_image(
        image,
        NORMALIZATION
    )

    return np.asarray(image, dtype=np.float32)


def calculate_average_image(paths, group_name):

    images = []

    print(f"\nProcessing {group_name}...")

    for path in paths:

        image = load_normalized_image(path)
        images.append(image)

    if not images:
        raise ValueError(f"No images found for {group_name}")

    average_image = np.mean(
        np.stack(images),
        axis=0
    )

    print(f"{group_name}: {len(images)} images processed")

    return average_image


def calculate_spatial_grid(average_image):

    cell_size = IMAGE_SIZE // GRID_SIZE

    cropped = average_image[
        :GRID_SIZE * cell_size,
        :GRID_SIZE * cell_size
    ]

    grid = cropped.reshape(
        GRID_SIZE,
        cell_size,
        GRID_SIZE,
        cell_size
    ).mean(axis=(1, 3))

    return grid


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

def plot_average_images(group_images):

    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 9)
    )

    axes = axes.flatten()

    for ax, (group_name, average_image) in zip(
        axes,
        group_images.items()
    ):

        ax.imshow(
            average_image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(group_name)
        ax.axis("off")

    fig.suptitle(
        "Average Normalized Screw Images",
        fontsize=16
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "spatial_average_images.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved: {output_path}")


def plot_spatial_grids(group_images):

    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 9)
    )

    axes = axes.flatten()

    for ax, (group_name, average_image) in zip(
        axes,
        group_images.items()
    ):

        grid = calculate_spatial_grid(average_image)

        image = ax.imshow(
            grid,
            cmap="viridis",
            vmin=0,
            vmax=255
        )

        ax.set_title(group_name)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

    fig.suptitle(
        "8x8 Spatial Brightness Feature Maps",
        fontsize=16
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "spatial_feature_grids.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(f"\nSaved: {output_path}")


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("\n========== SPATIAL VISUALIZATION ==========\n")

    training_paths = vm.collect_training_images()

    testing_paths, test_labels, test_categories = (
        vm.collect_testing_images()
    )

    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE
    )

    test_paths = np.asarray(testing_paths)
    test_categories = np.asarray(test_categories)

    # Identify normal test images
    normal_mask = np.array([
        str(category).strip().lower() in {"good", "normal"}
        for category in test_categories
    ])

    if not np.any(normal_mask):
        normal_mask = np.array([
            str(label).strip().lower() in {"0", "good", "normal"}
            for label in test_labels
        ])

    normal_paths = test_paths[normal_mask]
    defective_mask = ~normal_mask

    # Build groups
    groups = {
        "Training Fit Good": list(fit_paths),
        "Calibration Good": list(calibration_paths),
        "Test Good": list(normal_paths),
    }

    # Add each defect category
    defect_categories = sorted(
        set(test_categories[defective_mask])
    )

    for category in defect_categories:

        mask = (
            (test_categories == category)
            & defective_mask
        )

        groups[str(category)] = list(test_paths[mask])

    print("\n========== IMAGE COUNTS ==========\n")

    for name, paths in groups.items():
        print(f"{name}: {len(paths)}")

    # Calculate average images
    average_images = {}

    for name, paths in groups.items():

        average_images[name] = calculate_average_image(
            paths,
            name
        )

    # Generate visualizations
    print("\nGenerating average image comparison...")

    plot_average_images(average_images)

    print("\nGenerating spatial feature grids...")

    plot_spatial_grids(average_images)

    print("\n========== EXPERIMENT COMPLETE ==========\n")

    print("Two visualization files were generated.")
    print("No production model or saved model artifacts were changed.")


if __name__ == "__main__":
    main()
