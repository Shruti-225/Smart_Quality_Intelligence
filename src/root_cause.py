from pathlib import Path

import pandas as pd


def load_data(data_path: Path) -> pd.DataFrame:
    """Load cleaned production data."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    return pd.read_csv(data_path)


def analyze_root_causes(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Identify features associated with defective products."""

    if "quality_label" not in data.columns:
        raise ValueError(
            "quality_label column not found in dataset."
        )

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_columns = [
        column
        for column in numeric_columns
        if column != "quality_label"
    ]

    good_data = data[
        data["quality_label"] == 0
    ]

    defective_data = data[
        data["quality_label"] == 1
    ]

    results = []

    for feature in numeric_columns:

        good_mean = good_data[feature].mean()
        defective_mean = defective_data[feature].mean()

        difference = defective_mean - good_mean

        if good_mean != 0:
            percentage_change = (
                abs(difference) /
                abs(good_mean)
            ) * 100
        else:
            percentage_change = 0

        results.append({
            "feature": feature,
            "good_mean": good_mean,
            "defective_mean": defective_mean,
            "difference": difference,
            "percentage_change": percentage_change
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "percentage_change",
        ascending=False
    )

    return results_df


def save_results(
    results: pd.DataFrame,
    output_path: Path
) -> None:
    """Save root cause analysis results."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    data_path = (
        project_root
        / "data"
        / "processed"
        / "production_cleaned.csv"
    )

    output_path = (
        project_root
        / "outputs"
        / "reports"
        / "root_cause_analysis.csv"
    )

    print(
        "\n========== ROOT CAUSE ANALYSIS =========="
    )

    print("\nLoading production data...")

    data = load_data(
        data_path
    )

    print(
        f"Total records: {len(data)}"
    )

    print(
        f"Good records: "
        f"{(data['quality_label'] == 0).sum()}"
    )

    print(
        f"Defective records: "
        f"{(data['quality_label'] == 1).sum()}"
    )

    print(
        "\nAnalyzing feature differences..."
    )

    results = analyze_root_causes(
        data
    )

    print(
        "\n========== TOP ROOT CAUSES =========="
    )

    print(
        results.head(15).to_string(
            index=False
        )
    )

    save_results(
        results,
        output_path
    )

    print(
        "\nRoot cause report saved to:"
    )

    print(output_path)

    print(
        "\n========== ROOT CAUSE ANALYSIS COMPLETE =========="
    )