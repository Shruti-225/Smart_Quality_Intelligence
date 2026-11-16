
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import vision_model as vm
import normalization_experiment as ne


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

NORMALIZATION_METHOD = "Percentile"

# Use the same 98-feature configuration as our previous
# experiments. Indices 0-4 are intensity statistics.
FEATURE_INDICES = list(range(5, 103))

TOP_FEATURES = 10


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names():

    names = [
        f"intensity_stat_{i}"
        for i in range(5)
    ]

    names += [
        f"histogram_bin_{i}"
        for i in range(32)
    ]

    names += [
        "edge_density",
        "laplacian_variance",
    ]

    names += [
        f"spatial_mean_row_{row}_col_{col}"
        for row in range(8)
        for col in range(8)
    ]

    return names


# ============================================================
# FEATURE COMPARISON
# ============================================================

def analyze_category(
    feature_matrix,
    labels,
    feature_names,
    category,
):

    normal_mask = (
        (labels == 0)
    )

    defect_mask = (
        (labels == 1)
    )

    X_normal = feature_matrix[normal_mask]
    X_defect = feature_matrix[defect_mask]

    results = []

    for index, feature_name in enumerate(feature_names):

        normal_values = X_normal[:, index]
        defect_values = X_defect[:, index]

        normal_mean = np.mean(normal_values)
        defect_mean = np.mean(defect_values)

        normal_median = np.median(normal_values)
        defect_median = np.median(defect_values)

        pooled_std = np.sqrt(
            (
                np.var(normal_values) +
                np.var(defect_values)
            ) / 2
        )

        if pooled_std > 1e-12:
            effect_size = (
                defect_mean - normal_mean
            ) / pooled_std
        else:
            effect_size = 0.0

        combined_values = np.concatenate([
            normal_values,
            defect_values,
        ])

        combined_labels = np.concatenate([
            np.zeros(len(normal_values)),
            np.ones(len(defect_values)),
        ])

        try:
            raw_auc = roc_auc_score(
                combined_labels,
                combined_values,
            )

            separability_auc = max(
                raw_auc,
                1 - raw_auc,
            )

        except ValueError:
            raw_auc = 0.5
            separability_auc = 0.5

        results.append({
            "Category": category,
            "Feature": feature_name,
            "Feature Index": FEATURE_INDICES[index],
            "Normal Mean": normal_mean,
            "Defect Mean": defect_mean,
            "Normal Median": normal_median,
            "Defect Median": defect_median,
            "Effect Size": effect_size,
            "Raw AUC": raw_auc,
            "Separability AUC": separability_auc,
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n========== CATEGORY FEATURE ANALYSIS ==========")

    training_paths = vm.collect_training_images()

    testing_paths, test_labels, categories = (
        vm.collect_testing_images()
    )

    test_labels = np.asarray(test_labels)
    categories = np.asarray(categories)

    fit_paths, calibration_paths = train_test_split(
        training_paths,
        test_size=0.20,
        random_state=vm.RANDOM_STATE,
    )

    print("\nTraining fit:", len(fit_paths))
    print("Calibration:", len(calibration_paths))
    print("Testing:", len(testing_paths))

    # Extract normalized image features
    fit_features = ne.build_normalized_matrix(
        fit_paths,
        NORMALIZATION_METHOD,
        "training images",
    )

    calibration_features = ne.build_normalized_matrix(
        calibration_paths,
        NORMALIZATION_METHOD,
        "calibration images",
    )

    test_features = ne.build_normalized_matrix(
        testing_paths,
        NORMALIZATION_METHOD,
        "testing images",
    )

    # Select features
    test_features = test_features[:, FEATURE_INDICES]

    feature_names = get_feature_names()[5:103]

    print("\nSelected features:", len(feature_names))

    # Analyze each defect category against normal test images
    all_results = []

    defect_categories = sorted(
        set(categories[test_labels == 1])
    )

    for category in defect_categories:

        print("\n" + "=" * 65)
        print("ANALYZING:", category)
        print("=" * 65)

        category_mask = (
            (categories == category) &
            (test_labels == 1)
        )

        normal_mask = test_labels == 0

        selected_mask = normal_mask | category_mask

        category_features = test_features[selected_mask]
        category_labels = test_labels[selected_mask]

        results = analyze_category(
            category_features,
            category_labels,
            feature_names,
            category,
        )

        results = results.sort_values(
            "Separability AUC",
            ascending=False,
        )

        print("\nTop features by separability:")

        print(
            results.head(TOP_FEATURES).to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

        all_results.append(results)

    # Combine all category results
    final_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    print("\n========== TOP FEATURE SUMMARY ==========")

    summary = (
        final_results
        .sort_values(
            ["Category", "Separability AUC"],
            ascending=[True, False],
        )
        .groupby("Category")
        .head(TOP_FEATURES)
    )

    print(
        summary[
            [
                "Category",
                "Feature",
                "Feature Index",
                "Effect Size",
                "Raw AUC",
                "Separability AUC",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n========== EXPERIMENT COMPLETE ==========")
    print("No production model files were modified.")
    print("No model was retrained or saved.")


if __name__ == "__main__":
    main()
