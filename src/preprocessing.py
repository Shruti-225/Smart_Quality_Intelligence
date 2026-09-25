from pathlib import Path
import pandas as pd


def calculate_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing-value statistics for each column.
    """

    missing_count = data.isna().sum()
    missing_percentage = (missing_count / len(data)) * 100

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    return missing_report.sort_values(
        by="missing_percentage",
        ascending=False
    )


def remove_high_missing_columns(
    data: pd.DataFrame,
    threshold: float = 50.0
) -> pd.DataFrame:
    """
    Remove feature columns having missing values above
    the specified percentage threshold.
    """

    missing_percentage = data.isna().mean() * 100

    columns_to_remove = missing_percentage[
        missing_percentage > threshold
    ].index

    columns_to_remove = [
        column for column in columns_to_remove
        if column not in ["quality_label", "timestamp"]
    ]

    cleaned_data = data.drop(
        columns=columns_to_remove
    )

    print(
        f"Removed {len(columns_to_remove)} "
        f"columns with excessive missing values."
    )

    return cleaned_data


def remove_constant_columns(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove columns containing only one unique value.
    """

    feature_columns = [
        column
        for column in data.columns
        if column not in ["quality_label", "timestamp"]
    ]

    constant_columns = [
        column
        for column in feature_columns
        if data[column].nunique(dropna=True) <= 1
    ]

    cleaned_data = data.drop(
        columns=constant_columns
    )

    print(
        f"Removed {len(constant_columns)} "
        f"constant columns."
    )

    return cleaned_data


def impute_missing_values(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing numerical values using median imputation.
    """

    cleaned_data = data.copy()

    numeric_columns = cleaned_data.select_dtypes(
        include="number"
    ).columns

    numeric_feature_columns = [
        column
        for column in numeric_columns
        if column != "quality_label"
    ]

    for column in numeric_feature_columns:
        median_value = cleaned_data[column].median()

        cleaned_data[column] = cleaned_data[column].fillna(
            median_value
        )

    return cleaned_data


def convert_quality_label(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert SECOM labels into a business-friendly format.

    -1 = Good
     1 = Defective
    """

    cleaned_data = data.copy()

    cleaned_data["quality_label"] = (
        cleaned_data["quality_label"]
        .map({
            -1: 0,
            1: 1
        })
    )

    return cleaned_data


def preprocess_data(
    data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete preprocessing pipeline.
    """

    print("\n========== PREPROCESSING ==========")

    missing_report = calculate_missing_values(data)

    cleaned_data = remove_high_missing_columns(data)

    cleaned_data = remove_constant_columns(
        cleaned_data
    )

    cleaned_data = impute_missing_values(
        cleaned_data
    )

    cleaned_data = convert_quality_label(
        cleaned_data
    )

    return cleaned_data, missing_report


def save_processed_data(
    data: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save the processed dataset.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nProcessed dataset saved to:\n{output_path}"
    )


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    production_dir = (
        project_root
        / "data"
        / "raw"
        / "production"
    )

    processed_dir = (
        project_root
        / "data"
        / "processed"
    )

    data_file = production_dir / "secom.data"
    labels_file = production_dir / "secom_labels.data"

    production_data = pd.read_csv(
        data_file,
        sep=r"\s+",
        header=None
    )

    labels = pd.read_csv(
        labels_file,
        sep=r"\s+",
        header=None
    )

    production_data = pd.concat(
    [
        production_data,
        labels.rename(
            columns={
                0: "quality_label",
                1: "timestamp"
            }
        )
    ],
    axis=1
)

    processed_data, missing_report = preprocess_data(
        production_data
    )

    output_file = (
        processed_dir
        / "production_cleaned.csv"
    )

    report_file = (
        processed_dir
        / "missing_value_report.csv"
    )

    save_processed_data(
        processed_data,
        output_file
    )

    missing_report.to_csv(
        report_file,
        index=True
    )

    print("\n========== FINAL DATASET ==========")
    print(
        f"Rows: {processed_data.shape[0]}"
    )
    print(
        f"Columns: {processed_data.shape[1]}"
    )

    print("\nQuality distribution:")
    print(
        processed_data["quality_label"]
        .value_counts()
    )