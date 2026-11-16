
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import vision_model as vm
import normalization_experiment as ne


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 128
GRID_SIZE = 8
NORMALIZATION = "Percentile"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "vision"
)

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
        raise ValueError(f"Cannot read image: {image_path}")

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
        images.append(load_normalized_image(path))

    if not images:
        raise ValueError(f"No images found for {group_name}")

    return np.mean(
        np.stack(images),
        axis=0
    )


def calculate_spatial_grid(image):

    cell_size = IMAGE_SIZE // GRID_SIZE

    image = image[
        :GRID_SIZE * cell_size,
        :GRID_SIZE * cell_size
    ]

    grid = image.reshape(
        GRID_SIZE,
        cell_size,
        GRID_SIZE,
        cell_size
    ).mean(axis=(1, 3))

    return grid


# --------------------------------------------------
# DIFFERENCE ANALYSIS
# --------------------------------------------------

def calculate_region_statistics(difference):

    center = difference[2:6, 2:6]

    border_mask = np.ones(
        (GRID_SIZE, GRID_SIZE),
        dtype=bool
    )

    border_mask[2:6, 2:6] = False

    border = difference[border_mask]

    return {
        "Center Mean Difference": np.mean(center),
        "Border Mean Difference": np.mean(border),
        "Overall Mean Difference": np.mean(difference),
        "Maximum Difference": np.max(difference),
    }


def main():

    print("\n========== SPATIAL DIFFERENCE ANALYSIS ==========\n")

    # Collect image paths
    training_paths = vm.collect_training_images()

    testing_paths, test_labels, test_categories = (
        vm.collect_testing_images()
    )

    # Preserve existing split
    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE
    )

    test_paths = np.asarray(testing_paths)
    test_categories = np.asarray(test_categories)
    test_labels = np.asarray(test_labels)

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

    defective_mask = ~normal_mask

    # Create image groups
    groups = {
        "Training Fit Good": list(fit_paths),
        "Test Good": list(test_paths[normal_mask]),
    }

    for category in sorted(set(test_categories[defective_mask])):

        mask = (
            (test_categories == category)
            & defective_mask
        )

        groups[str(category)] = list(test_paths[mask])

    print("\n========== IMAGE COUNTS ==========\n")

    for name, paths in groups.items():
        print(f"{name}: {len(paths)}")

    # Calculate average images and spatial grids
    spatial_grids = {}

    for name, paths in groups.items():

        average_image = calculate_average_image(
            paths,
            name
        )

        spatial_grids[name] = calculate_spatial_grid(
            average_image
        )

    # Training Good is our reference
    baseline = spatial_grids["Training Fit Good"]

    differences = {}

    for name, grid in spatial_grids.items():

        if name == "Training Fit Good":
            continue

        differences[name] = grid - baseline

    # --------------------------------------------------
    # PRINT CELL DIFFERENCES
    # --------------------------------------------------

    print("\n========== CELL DIFFERENCE ANALYSIS ==========\n")

    csv_rows = []

    for group_name, difference in differences.items():

        print(f"\n--- {group_name} ---")

        absolute_difference = np.abs(difference)

        flat_indices = np.argsort(
            absolute_difference.ravel()
        )[::-1]

        print("Top 5 changed cells:")

        for index in flat_indices[:5]:

            row, col = np.unravel_index(
                index,
                difference.shape
            )

            signed_value = difference[row, col]

            print(
                f"Row {row}, Col {col}: "
                f"Difference = {signed_value:+.3f}"
            )

        stats = calculate_region_statistics(
            absolute_difference
        )

        print("\nRegional statistics:")

        for key, value in stats.items():
            print(f"{key}: {value:.3f}")

        # Store all 64 cell differences
        for row in range(GRID_SIZE):

            for col in range(GRID_SIZE):

                csv_rows.append({
                    "Group": group_name,
                    "Row": row,
                    "Column": col,
                    "Baseline Brightness": baseline[row, col],
                    "Group Brightness": spatial_grids[group_name][row, col],
                    "Signed Difference": difference[row, col],
                    "Absolute Difference": absolute_difference[row, col],
                })

    # --------------------------------------------------
    # SAVE CSV REPORT
    # --------------------------------------------------

    report = pd.DataFrame(csv_rows)

    csv_path = OUTPUT_DIR / "spatial_cell_differences.csv"

    report.to_csv(
        csv_path,
        index=False
    )

    print(f"\nCSV report saved: {csv_path}")

    # --------------------------------------------------
    # DIFFERENCE HEATMAPS
    # --------------------------------------------------

    max_difference = max(
        np.max(np.abs(diff))
        for diff in differences.values()
    )

    max_difference = max(max_difference, 1)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    axes = axes.flatten()

    for ax, (group_name, difference) in zip(
        axes,
        differences.items()
    ):

        image = ax.imshow(
            difference,
            cmap="coolwarm",
            vmin=-max_difference,
            vmax=max_difference
        )

        ax.set_title(group_name)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        for row in range(GRID_SIZE):

            for col in range(GRID_SIZE):

                ax.text(
                    col,
                    row,
                    f"{difference[row, col]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6
                )

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

    fig.suptitle(
        "Spatial Brightness Difference vs Training Good",
        fontsize=15
    )

    plt.tight_layout()

    heatmap_path = (
        OUTPUT_DIR
        / "spatial_difference_heatmaps.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nHeatmaps saved: {heatmap_path}")

    print("\n========== EXPERIMENT COMPLETE ==========\n")
    print("No production model or saved model artifacts were changed.")


if __name__ == "__main__":
    main()
