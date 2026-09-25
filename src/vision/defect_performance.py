from pathlib import Path

import pandas as pd


def get_project_root():
    """Return the project root directory."""

    return Path(__file__).resolve().parents[2]


def analyze_defect_performance():
    """Analyze Vision model performance for each defect category."""

    project_root = get_project_root()

    prediction_file = (
        project_root
        / "outputs"
        / "reports"
        / "vision"
        / "vision_predictions.csv"
    )

    output_directory = (
        project_root
        / "outputs"
        / "reports"
        / "vision"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== DEFECT PERFORMANCE ANALYSIS ==========")

    if not prediction_file.exists():

        print(
            f"\nERROR: Prediction file not found:"
            f"\n{prediction_file}"
        )

        return

    data = pd.read_csv(prediction_file)

    required_columns = {
        "image",
        "category",
        "actual_label",
        "predicted_label",
        "anomaly_score",
    }

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:

        print(
            "\nERROR: Missing required columns:"
        )

        for column in missing_columns:
            print(f"- {column}")

        return

    results = []

    categories = data["category"].unique()

    for category in categories:

        category_data = data[
            data["category"] == category
        ]

        total_images = len(category_data)

        correctly_classified = (
            category_data["actual_label"]
            == category_data["predicted_label"]
        ).sum()

        incorrectly_classified = (
            total_images
            - correctly_classified
        )

        detection_rate = (
            correctly_classified
            / total_images
            if total_images > 0
            else 0
        )

        average_anomaly_score = (
            category_data["anomaly_score"].mean()
        )

        minimum_anomaly_score = (
            category_data["anomaly_score"].min()
        )

        maximum_anomaly_score = (
            category_data["anomaly_score"].max()
        )

        results.append(
            {
                "category": category,
                "total_images": total_images,
                "correctly_classified": correctly_classified,
                "incorrectly_classified": incorrectly_classified,
                "detection_rate": detection_rate,
                "average_anomaly_score":
                    average_anomaly_score,
                "minimum_anomaly_score":
                    minimum_anomaly_score,
                "maximum_anomaly_score":
                    maximum_anomaly_score,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="detection_rate"
    )

    output_file = (
        output_directory
        / "defect_performance.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n========== RESULTS ==========")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nReport saved to:"
        f"\n{output_file}"
    )

    print(
        "\n========== MOST DIFFICULT CATEGORY =========="
    )

    if not results_df.empty:

        most_difficult = (
            results_df.iloc[0]
        )

        print(
            f"Category: "
            f"{most_difficult['category']}"
        )

        print(
            f"Detection Rate: "
            f"{most_difficult['detection_rate']:.2%}"
        )

        print(
            f"Incorrectly Classified: "
            f"{int(most_difficult['incorrectly_classified'])}"
        )


if __name__ == "__main__":

    analyze_defect_performance()