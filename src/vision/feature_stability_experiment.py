from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import vision_model as vm
import normalization_experiment as ne


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

FEATURES = {
    95: "spatial_mean_row_7_col_0",
    39: "spatial_mean_row_0_col_0",
    16: "histogram_bin_11",
    87: "spatial_mean_row_6_col_0",
    96: "spatial_mean_row_7_col_1",
    97: "spatial_mean_row_7_col_2",
    98: "spatial_mean_row_7_col_3",
    9: "histogram_bin_4",
}

NORMALIZATION = "Percentile"


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_statistics(values):
    values = np.asarray(values, dtype=float)

    return {
        "Count": len(values),
        "Mean": np.mean(values),
        "Median": np.median(values),
        "Std": np.std(values),
        "Q1": np.percentile(values, 25),
        "Q3": np.percentile(values, 75),
        "Min": np.min(values),
        "Max": np.max(values),
    }


def calculate_effect_size(group_a, group_b):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    pooled_std = np.sqrt(
        (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2
    )

    if pooled_std == 0:
        return 0.0

    return (np.mean(b) - np.mean(a)) / pooled_std


def calculate_separability(group_a, group_b):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    labels = np.concatenate([
        np.zeros(len(a)),
        np.ones(len(b))
    ])

    values = np.concatenate([a, b])

    auc = roc_auc_score(labels, values)

    return max(auc, 1 - auc)


def is_normal_image(label, category):
    label_text = str(label).strip().lower()
    category_text = str(category).strip().lower()

    if "good" in category_text or "normal" in category_text:
        return True

    if label_text in {"0", "good", "normal"}:
        return True

    return False


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    print("\n========== FEATURE STABILITY ANALYSIS ==========\n")

    # Collect image paths
    training_paths = vm.collect_training_images()

    testing_paths, test_labels, test_categories = (
        vm.collect_testing_images()
    )

    # Preserve the existing 256 / 64 split
    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE
    )

    test_labels = np.asarray(test_labels)
    test_categories = np.asarray(test_categories)

    # Identify normal and defective test images
    normal_mask = np.array([
        is_normal_image(label, category)
        for label, category in zip(test_labels, test_categories)
    ])

    defective_mask = ~normal_mask

    normal_test_paths = np.asarray(testing_paths)[normal_mask]
    defective_test_paths = np.asarray(testing_paths)[defective_mask]

    print(f"Training fit Good: {len(fit_paths)}")
    print(f"Calibration Good: {len(calibration_paths)}")
    print(f"Test Good: {len(normal_test_paths)}")
    print(f"Test Defective: {len(defective_test_paths)}")

    # Extract features once using the same normalization
    all_paths = (
        list(fit_paths)
        + list(calibration_paths)
        + list(normal_test_paths)
        + list(defective_test_paths)
    )

    print("\nExtracting normalized features...")

    all_features = ne.build_normalized_matrix(
        all_paths,
        NORMALIZATION,
        "Feature Stability Analysis"
    )

    all_features = np.asarray(all_features)

    # Split feature matrix back into groups
    n_fit = len(fit_paths)
    n_cal = len(calibration_paths)
    n_normal = len(normal_test_paths)

    fit_features = all_features[:n_fit]

    calibration_features = all_features[
        n_fit:n_fit + n_cal
    ]

    normal_features = all_features[
        n_fit + n_cal:n_fit + n_cal + n_normal
    ]

    defective_features = all_features[
        n_fit + n_cal + n_normal:
    ]

    # Prepare category-specific feature groups
    category_groups = {}

    for category in sorted(set(test_categories[defective_mask])):
        category_mask = (
            (test_categories == category) & defective_mask
        )

        category_paths = np.asarray(testing_paths)[category_mask]

        category_features = ne.build_normalized_matrix(
            list(category_paths),
            NORMALIZATION,
            f"Category: {category}"
        )

        category_groups[str(category)] = np.asarray(category_features)

    groups = {
        "Training Fit Good": fit_features,
        "Calibration Good": calibration_features,
        "Test Good": normal_features,
        **category_groups,
    }

    # -----------------------------------------------------
    # FEATURE DISTRIBUTION REPORT
    # -----------------------------------------------------

    print("\n========== FEATURE DISTRIBUTIONS ==========\n")

    rows = []

    for index, feature_name in FEATURES.items():

        for group_name, matrix in groups.items():

            values = matrix[:, index]

            stats = get_statistics(values)

            rows.append({
                "Feature": feature_name,
                "Group": group_name,
                **stats
            })

    report = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")

    print(report.to_string(index=False))

    # -----------------------------------------------------
    # TRAINING GOOD vs TEST GOOD
    # -----------------------------------------------------

    print("\n========== TRAINING GOOD vs TEST GOOD ==========\n")

    for index, feature_name in FEATURES.items():

        training_values = fit_features[:, index]
        test_values = normal_features[:, index]

        auc = calculate_separability(
            training_values,
            test_values
        )

        effect = calculate_effect_size(
            training_values,
            test_values
        )

        print(f"\nFeature: {feature_name}")
        print(f"Separability AUC: {auc:.5f}")
        print(f"Effect size: {effect:+.5f}")
        print(
            f"Training mean: {np.mean(training_values):.5f}"
        )
        print(
            f"Test Good mean: {np.mean(test_values):.5f}"
        )

    # -----------------------------------------------------
    # TEST GOOD vs EACH DEFECT CATEGORY
    # -----------------------------------------------------

    print("\n========== TEST GOOD vs DEFECT CATEGORIES ==========\n")

    for category, matrix in category_groups.items():

        print(f"\n--- {category} ---")

        for index, feature_name in FEATURES.items():

            good_values = normal_features[:, index]
            defect_values = matrix[:, index]

            auc = calculate_separability(
                good_values,
                defect_values
            )

            effect = calculate_effect_size(
                good_values,
                defect_values
            )

            print(
                f"{feature_name}: "
                f"AUC={auc:.5f}, "
                f"Effect={effect:+.5f}"
            )

    print("\n========== EXPERIMENT COMPLETE ==========\n")
    print("No production model or saved model artifacts were changed.")


if __name__ == "__main__":
    main()