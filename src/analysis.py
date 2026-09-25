from pathlib import Path

import pandas as pd


def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Load the cleaned production dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)


def summarize_quality(data: pd.DataFrame) -> pd.Series:
    """Return the distribution of quality labels."""

    return data["quality_label"].value_counts()


def calculate_feature_statistics(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Calculate descriptive statistics for numeric features."""

    feature_columns = [
        column
        for column in data.columns
        if column not in ["quality_label", "timestamp"]
    ]

    statistics = data[feature_columns].describe().T

    statistics["missing_values"] = (
        data[feature_columns].isna().sum()
    )

    return statistics


def compare_quality_groups(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Compare feature means between good and defective products."""

    feature_columns = [
        column
        for column in data.columns
        if column not in ["quality_label", "timestamp"]
    ]

    comparison = data.groupby(
        "quality_label"
    )[feature_columns].mean().T

    comparison.columns = [
        "Good" if column == 0 else "Defective"
        for column in comparison.columns
    ]

    comparison["absolute_difference"] = (
        comparison["Defective"] - comparison["Good"]
    ).abs()

    return comparison.sort_values(
        "absolute_difference",
        ascending=False
    )


def create_management_summary(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Create high-level quality management KPIs."""

    total_units = len(data)

    defective_units = int(
        (data["quality_label"] == 1).sum()
    )

    good_units = int(
        (data["quality_label"] == 0).sum()
    )

    defect_rate = (
        defective_units / total_units
        if total_units > 0
        else 0
    )

    summary = pd.DataFrame({
        "metric": [
            "Total Production Units",
            "Good Units",
            "Defective Units",
            "Defect Rate"
        ],
        "value": [
            total_units,
            good_units,
            defective_units,
            defect_rate
        ]
    })

    return summary


def load_existing_report(
    file_path: Path
) -> pd.DataFrame | None:
    """Load an existing report if available."""

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


def create_consolidated_report(
    project_root: Path,
    data: pd.DataFrame
) -> pd.DataFrame:
    """Combine existing project results into one report."""

    reports_dir = (
        project_root
        / "outputs"
        / "reports"
    )

    total_units = len(data)

    defective_units = int(
        (data["quality_label"] == 1).sum()
    )

    good_units = int(
        (data["quality_label"] == 0).sum()
    )

    defect_rate = (
        defective_units / total_units
        if total_units > 0
        else 0
    )

    metrics = []

    metrics.append({
        "category": "Production",
        "metric": "Total Production Units",
        "value": total_units
    })

    metrics.append({
        "category": "Production",
        "metric": "Good Units",
        "value": good_units
    })

    metrics.append({
        "category": "Production",
        "metric": "Defective Units",
        "value": defective_units
    })

    metrics.append({
        "category": "Quality",
        "metric": "Defect Rate",
        "value": defect_rate
    })

    model_report = load_existing_report(
        reports_dir / "model_comparison.csv"
    )

    if model_report is not None and not model_report.empty:

        best_model = model_report.sort_values(
            "f1_score",
            ascending=False
        ).iloc[0]

        metrics.append({
            "category": "ML Model",
            "metric": "Best Model",
            "value": best_model["model"]
        })

        metrics.append({
            "category": "ML Model",
            "metric": "Best Model F1 Score",
            "value": best_model["f1_score"]
        })

        metrics.append({
            "category": "ML Model",
            "metric": "Best Model ROC-AUC",
            "value": best_model["roc_auc"]
        })

    vision_report = load_existing_report(
        reports_dir
        / "vision"
        / "defect_performance.csv"
    )

    if vision_report is not None and not vision_report.empty:

        average_detection_rate = (
            vision_report["detection_rate"].mean()
        )

        difficult_category = (
            vision_report.sort_values(
                "detection_rate"
            ).iloc[0]
        )

        metrics.append({
            "category": "Vision",
            "metric": "Average Detection Rate",
            "value": average_detection_rate
        })

        metrics.append({
            "category": "Vision",
            "metric": "Most Difficult Defect",
            "value": difficult_category["category"]
        })

        metrics.append({
            "category": "Vision",
            "metric": "Difficult Defect Detection Rate",
            "value": difficult_category[
                "detection_rate"
            ]
        })

    business_report = load_existing_report(
        reports_dir / "business_impact.csv"
    )

    if business_report is not None and not business_report.empty:

        business = business_report.iloc[0]

        metrics.append({
            "category": "Business Impact",
            "metric": "Estimated Quality Loss",
            "value": business[
                "estimated_quality_loss"
            ]
        })

        metrics.append({
            "category": "Business Impact",
            "metric": "Potential Savings",
            "value": business[
                "potential_savings"
            ]
        })

    return pd.DataFrame(metrics)


def save_report(
    report: pd.DataFrame,
    output_path: Path
) -> None:
    """Save an analysis report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
    output_path,
    index=True,
    index_label="feature"
)


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    processed_file = (
        project_root
        / "data"
        / "processed"
        / "production_cleaned.csv"
    )

    report_directory = (
        project_root
        / "outputs"
        / "reports"
    )

    print(
        "\n========== CONSOLIDATED QUALITY ANALYSIS =========="
    )

    print(
        "\nLoading production data..."
    )

    data = load_processed_data(
        processed_file
    )

    print(
        f"Total records: {len(data)}"
    )

    print(
        "\n========== QUALITY DISTRIBUTION =========="
    )

    print(
        summarize_quality(data)
    )

    print(
        "\n========== DATASET SHAPE =========="
    )

    print(
        f"Rows: {data.shape[0]}"
    )

    print(
        f"Columns: {data.shape[1]}"
    )

    statistics = calculate_feature_statistics(
        data
    )

    save_report(
        statistics,
        report_directory
        / "feature_statistics.csv"
    )

    comparison = compare_quality_groups(
        data
    )

    save_report(
        comparison,
        report_directory
        / "quality_feature_comparison.csv"
    )

    print(
        "\n========== TOP DIFFERENT FEATURES =========="
    )

    print(
        comparison.head(15)
    )

    print(
        "\n========== MANAGEMENT SUMMARY =========="
    )

    management_summary = create_management_summary(
        data
    )

    print(
        management_summary.to_string(
            index=False
        )
    )

    consolidated_report = create_consolidated_report(
        project_root,
        data
    )

    consolidated_path = (
        report_directory
        / "consolidated_quality_analysis.csv"
    )

    save_report(
        consolidated_report,
        consolidated_path
    )

    print(
        "\n========== CONSOLIDATED RESULTS =========="
    )

    print(
        consolidated_report.to_string(
            index=False
        )
    )

    print(
        "\nConsolidated report saved to:"
    )

    print(
        consolidated_path
    )

    print(
        "\n========== ANALYSIS COMPLETE =========="
    )