from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_output_directory(output_dir: Path) -> None:
    """Create visualization output directory."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


def save_chart(output_dir: Path, filename: str) -> None:
    """Save and close the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_quality_distribution(
    data: pd.DataFrame,
    output_dir: Path
) -> None:
    """Good vs defective production records."""

    counts = data["quality_label"].value_counts()

    labels = ["Good", "Defective"]
    values = [
        counts.get(0, 0),
        counts.get(1, 0)
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(labels, values)

    plt.title("Production Quality Distribution")
    plt.xlabel("Quality Status")
    plt.ylabel("Number of Production Records")

    save_chart(
        output_dir,
        "quality_distribution.png"
    )


def plot_model_comparison(
    project_root: Path,
    output_dir: Path
) -> None:
    """Compare ML model performance."""

    file_path = (
        project_root
        / "outputs"
        / "reports"
        / "model_comparison.csv"
    )

    if not file_path.exists():
        print("Model comparison report not found.")
        return

    data = pd.read_csv(file_path)

    metrics = [
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in data.columns
    ]

    ax = data.set_index("model")[
        available_metrics
    ].plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("Machine Learning Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)

    save_chart(
        output_dir,
        "model_comparison.png"
    )


def plot_vision_performance(
    project_root: Path,
    output_dir: Path
) -> None:
    """Visualize vision detection rate by defect category."""

    file_path = (
        project_root
        / "outputs"
        / "reports"
        / "vision"
        / "defect_performance.csv"
    )

    if not file_path.exists():
        print("Vision performance report not found.")
        return

    data = pd.read_csv(file_path)

    data = data.sort_values(
        "detection_rate",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        data["category"],
        data["detection_rate"]
    )

    plt.title(
        "Vision Defect Detection Rate by Category"
    )

    plt.xlabel("Detection Rate")
    plt.ylabel("Defect Category")

    plt.xlim(0, 1.05)

    save_chart(
        output_dir,
        "vision_detection_rate.png"
    )


def plot_root_causes(
    project_root: Path,
    output_dir: Path
) -> None:
    """Visualize the strongest root-cause features."""

    file_path = (
        project_root
        / "outputs"
        / "reports"
        / "root_cause_analysis.csv"
    )

    if not file_path.exists():
        print("Root cause report not found.")
        return

    data = pd.read_csv(file_path)

    data = data.head(10)

    data = data.sort_values(
        "difference"
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        data["feature"].astype(str),
        data["difference"].abs()
    )

    plt.title(
        "Top 10 Potential Root-Cause Features"
    )

    plt.xlabel(
        "Absolute Difference Between Good and Defective"
    )

    plt.ylabel("Feature")

    save_chart(
        output_dir,
        "top_root_causes.png"
    )


def plot_business_impact(
    project_root: Path,
    output_dir: Path
) -> None:
    """Visualize business quality loss and potential savings."""

    file_path = (
        project_root
        / "outputs"
        / "reports"
        / "business_impact.csv"
    )

    if not file_path.exists():
        print("Business impact report not found.")
        return

    data = pd.read_csv(file_path)

    if "metric" not in data.columns:
        print("Business impact format not recognized.")
        return

    values = {}

    for _, row in data.iterrows():
        values[str(row["metric"])] = row["value"]

    quality_loss = values.get(
        "Estimated Quality Loss",
        0
    )

    potential_savings = values.get(
        "Potential Savings",
        0
    )

    labels = [
        "Estimated Quality Loss",
        "Potential Savings"
    ]

    amounts = [
        quality_loss,
        potential_savings
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        amounts
    )

    plt.title("Business Impact of Quality Defects")
    plt.ylabel("Amount (₹)")

    plt.xticks(rotation=10)

    save_chart(
        output_dir,
        "business_impact.png"
    )


def plot_anomaly_scores(
    project_root: Path,
    output_dir: Path
) -> None:
    """Visualize vision anomaly scores."""

    file_path = (
        project_root
        / "outputs"
        / "reports"
        / "vision"
        / "vision_predictions.csv"
    )

    if not file_path.exists():
        print("Vision predictions not found.")
        return

    data = pd.read_csv(file_path)

    good_scores = data.loc[
        data["actual_label"] == 0,
        "anomaly_score"
    ]

    defective_scores = data.loc[
        data["actual_label"] == 1,
        "anomaly_score"
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        good_scores,
        bins=20,
        alpha=0.7,
        label="Good"
    )

    plt.hist(
        defective_scores,
        bins=20,
        alpha=0.7,
        label="Defective"
    )

    plt.title(
        "Vision Anomaly Score Distribution"
    )

    plt.xlabel("Anomaly Score")
    plt.ylabel("Number of Images")

    plt.legend()

    save_chart(
        output_dir,
        "anomaly_score_distribution.png"
    )


def plot_feature_comparison(
    data: pd.DataFrame,
    output_dir: Path,
    feature_id: int
) -> None:
    """Compare a process feature between quality groups."""

    feature_column = str(feature_id)

    if feature_column not in data.columns:
        print(
            f"Feature {feature_id} not found."
        )
        return

    good_values = data.loc[
        data["quality_label"] == 0,
        feature_column
    ]

    defective_values = data.loc[
        data["quality_label"] == 1,
        feature_column
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [
            good_values,
            defective_values
        ],
        tick_labels=[
            "Good",
            "Defective"
        ]
    )

    plt.title(
        f"Feature {feature_id}: "
        "Good vs Defective"
    )

    plt.xlabel("Quality Status")
    plt.ylabel(
        f"Feature {feature_id}"
    )

    save_chart(
        output_dir,
        f"feature_{feature_id}_comparison.png"
    )


def plot_top_features(
    data: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create charts for important process features."""

    feature_ids = [
        161,
        159,
        21,
        24,
        160
    ]

    for feature_id in feature_ids:
        plot_feature_comparison(
            data,
            output_dir,
            feature_id
        )


if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    processed_file = (
        project_root
        / "data"
        / "processed"
        / "production_cleaned.csv"
    )

    output_dir = (
        project_root
        / "outputs"
        / "figures"
    )

    print(
        "\n========== VISUALIZATION PIPELINE =========="
    )

    print(
        "\nLoading production data..."
    )

    if not processed_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {processed_file}"
        )

    data = pd.read_csv(
        processed_file
    )

    print(
        f"Production records: {len(data)}"
    )

    create_output_directory(
        output_dir
    )

    print(
        "\nCreating quality distribution..."
    )

    plot_quality_distribution(
        data,
        output_dir
    )

    print(
        "Creating ML model comparison..."
    )

    plot_model_comparison(
        project_root,
        output_dir
    )

    print(
        "Creating vision performance chart..."
    )

    plot_vision_performance(
        project_root,
        output_dir
    )

    print(
        "Creating root-cause chart..."
    )

    plot_root_causes(
        project_root,
        output_dir
    )

    print(
        "Creating business impact chart..."
    )

    plot_business_impact(
        project_root,
        output_dir
    )

    print(
        "Creating anomaly-score chart..."
    )

    plot_anomaly_scores(
        project_root,
        output_dir
    )

    print(
        "Creating feature comparison charts..."
    )

    plot_top_features(
        data,
        output_dir
    )

    print(
        "\n========== VISUALIZATION COMPLETE =========="
    )

    print(
        f"Charts saved to:\n{output_dir}"
    )