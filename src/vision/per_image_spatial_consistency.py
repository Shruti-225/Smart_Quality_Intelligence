
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
CALIBRATION_PERCENTILE = 95

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


def extract_grids(paths, group_name):

    grids = []

    print(f"\nProcessing {group_name}...")

    for index, path in enumerate(paths, start=1):

        image = load_normalized_image(path)

        grid = calculate_spatial_grid(image)

        grids.append(grid)

        if index % 50 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)}")

    return np.asarray(grids)


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("\n========== PER-IMAGE SPATIAL CONSISTENCY ==========\n")

    # Collect dataset
    training_paths = vm.collect_training_images()

    testing_paths, test_labels, test_categories = (
        vm.collect_testing_images()
    )

    # Preserve existing training split
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

    normal_paths = list(test_paths[normal_mask])
    defective_paths = list(test_paths[defective_mask])

    print("\n========== IMAGE COUNTS ==========\n")
    print(f"Training Fit Good: {len(fit_paths)}")
    print(f"Calibration Good: {len(calibration_paths)}")
    print(f"Test Good: {len(normal_paths)}")
    print(f"Test Defective: {len(defective_paths)}")

    # Extract individual spatial grids
    fit_grids = extract_grids(
        list(fit_paths),
        "Training Fit Good"
    )

    calibration_grids = extract_grids(
        list(calibration_paths),
        "Calibration Good"
    )

    normal_grids = extract_grids(
        normal_paths,
        "Test Good"
    )

    defective_grids = extract_grids(
        defective_paths,
        "Test Defective"
    )

    # --------------------------------------------------
    # ESTABLISH NORMAL REFERENCE
    # --------------------------------------------------

    print("\n========== CALIBRATION ==========\n")

    # Reference is calculated only from Training Fit Good
    baseline_grid = np.mean(
        fit_grids,
        axis=0
    )

    # Calibration deviations from the training reference
    calibration_deviations = np.abs(
        calibration_grids - baseline_grid
    )

    # Cell-specific limits learned from calibration Good
    cell_thresholds = np.percentile(
        calibration_deviations,
        CALIBRATION_PERCENTILE,
        axis=0
    )

    # Fraction of cells exceeding their individual limits
    calibration_scores = np.mean(
        calibration_deviations > cell_thresholds,
        axis=(1, 2)
    )

    # Image-level limit derived from calibration Good
    image_threshold = np.percentile(
        calibration_scores,
        CALIBRATION_PERCENTILE
    )

    print(
        f"Cell threshold percentile: "
        f"{CALIBRATION_PERCENTILE}"
    )

    print(
        f"Image anomaly threshold: "
        f"{image_threshold:.5f}"
    )

    print(
        f"Calibration mean anomaly fraction: "
        f"{np.mean(calibration_scores):.5f}"
    )

    # --------------------------------------------------
    # EVALUATE INDIVIDUAL IMAGES
    # --------------------------------------------------

    def evaluate_group(grids, group_name):

        deviations = np.abs(
            grids - baseline_grid
        )

        exceedance = deviations > cell_thresholds

        scores = np.mean(
            exceedance,
            axis=(1, 2)
        )

        predictions = scores > image_threshold

        return {
            "name": group_name,
            "deviations": deviations,
            "exceedance": exceedance,
            "scores": scores,
            "predictions": predictions,
        }

    results = {}

    results["Test Good"] = evaluate_group(
        normal_grids,
        "Test Good"
    )

    results["Test Defective"] = evaluate_group(
        defective_grids,
        "Test Defective"
    )

    # --------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------

    print("\n========== OVERALL RESULTS ==========\n")

    summary_rows = []

    for group_name, result in results.items():

        scores = result["scores"]
        predictions = result["predictions"]

        flagged = int(np.sum(predictions))

        detection_rate = (
            np.mean(predictions) * 100
        )

        print(f"\n{group_name}")

        print(f"Images: {len(scores)}")
        print(f"Flagged images: {flagged}")
        print(f"Flag rate: {detection_rate:.2f}%")
        print(f"Mean anomaly fraction: {np.mean(scores):.5f}")
        print(f"Median anomaly fraction: {np.median(scores):.5f}")
        print(f"Maximum anomaly fraction: {np.max(scores):.5f}")

        summary_rows.append({
            "Group": group_name,
            "Images": len(scores),
            "Flagged": flagged,
            "Flag Rate (%)": detection_rate,
            "Mean Anomaly Fraction": np.mean(scores),
            "Median Anomaly Fraction": np.median(scores),
            "Maximum Anomaly Fraction": np.max(scores),
        })

    # --------------------------------------------------
    # CATEGORY-WISE ANALYSIS
    # --------------------------------------------------

    print("\n========== DEFECT CATEGORY CONSISTENCY ==========\n")

    category_rows = []
    category_cell_maps = {}

    for category in sorted(
        set(test_categories[defective_mask])
    ):

        category_mask = (
            (test_categories == category)
            & defective_mask
        )

        category_paths = list(test_paths[category_mask])

        category_grids = extract_grids(
            category_paths,
            str(category)
        )

        category_result = evaluate_group(
            category_grids,
            str(category)
        )

        scores = category_result["scores"]
        exceedance = category_result["exceedance"]

        cell_frequency = np.mean(
            exceedance,
            axis=0
        )

        category_cell_maps[str(category)] = cell_frequency

        flagged = int(
            np.sum(category_result["predictions"])
        )

        print(f"\n{category}")
        print(f"Images: {len(scores)}")
        print(f"Flagged: {flagged}")
        print(f"Flag rate: {np.mean(category_result['predictions']) * 100:.2f}%")
        print(f"Mean anomaly fraction: {np.mean(scores):.5f}")

        category_rows.append({
            "Category": str(category),
            "Images": len(scores),
            "Flagged": flagged,
            "Flag Rate (%)": np.mean(
                category_result["predictions"]
            ) * 100,
            "Mean Anomaly Fraction": np.mean(scores),
            "Median Anomaly Fraction": np.median(scores),
        })

    # --------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    category_df = pd.DataFrame(category_rows)

    summary_path = (
        OUTPUT_DIR
        / "per_image_spatial_summary.csv"
    )

    category_path = (
        OUTPUT_DIR
        / "per_image_category_consistency.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    category_df.to_csv(
        category_path,
        index=False
    )

    # --------------------------------------------------
    # SAVE CELL CONSISTENCY REPORT
    # --------------------------------------------------

    cell_rows = []

    all_cell_maps = {
        "Test Good": np.mean(
            results["Test Good"]["exceedance"],
            axis=0
        ),
        **category_cell_maps,
    }

    for group_name, cell_map in all_cell_maps.items():

        for row in range(GRID_SIZE):

            for col in range(GRID_SIZE):

                cell_rows.append({
                    "Group": group_name,
                    "Row": row,
                    "Column": col,
                    "Exceedance Frequency": cell_map[row, col],
                    "Exceedance Percentage": cell_map[row, col] * 100,
                })

    cell_df = pd.DataFrame(cell_rows)

    cell_path = (
        OUTPUT_DIR
        / "per_image_spatial_cell_consistency.csv"
    )

    cell_df.to_csv(
        cell_path,
        index=False
    )

    # --------------------------------------------------
    # VISUALIZE CONSISTENCY MAPS
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    axes = axes.flatten()

    for ax, (group_name, cell_map) in zip(
        axes,
        all_cell_maps.items()
    ):

        image = ax.imshow(
            cell_map * 100,
            cmap="magma",
            vmin=0,
            vmax=100
        )

        ax.set_title(group_name)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Images exceeding limit (%)"
        )

    fig.suptitle(
        "Per-Image Spatial Consistency Maps",
        fontsize=15
    )

    plt.tight_layout()

    heatmap_path = (
        OUTPUT_DIR
        / "per_image_spatial_consistency.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\n========== REPORTS SAVED ==========\n")

    print(summary_path)
    print(category_path)
    print(cell_path)
    print(heatmap_path)

    print("\n========== EXPERIMENT COMPLETE ==========\n")
    print("Production model and saved model artifacts were not changed.")


if __name__ == "__main__":
    main()
