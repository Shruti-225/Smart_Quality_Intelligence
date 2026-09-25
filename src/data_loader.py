from pathlib import Path
import pandas as pd


def load_production_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load SECOM production measurements and quality labels.

    Parameters
    ----------
    data_dir : Path
        Directory containing secom.data and secom_labels.data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Production measurements and quality labels.
    """

    data_file = data_dir / "secom.data"
    labels_file = data_dir / "secom_labels.data"

    if not data_file.exists():
        raise FileNotFoundError(f"Production data not found: {data_file}")

    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

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

    return production_data, labels


def combine_production_data(
    production_data: pd.DataFrame,
    labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine SECOM process measurements with quality labels.
    """

    result = production_data.copy()

    result["quality_label"] = labels.iloc[:, 0].values
    result["timestamp"] = labels.iloc[:, 1].values

    return result


def inspect_dataset(data: pd.DataFrame) -> None:
    """
    Print basic information about the production dataset.
    """

    print("\n========== DATASET INFORMATION ==========")

    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nData types:")
    print(data.dtypes.value_counts())

    print("\nMissing values:")
    print(data.isna().sum().sum())

    print("\nQuality label distribution:")
    print(data["quality_label"].value_counts())


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    production_dir = project_root / "data" / "raw" / "production"

    production_data, labels = load_production_data(production_dir)

    combined_data = combine_production_data(
        production_data,
        labels
    )

    inspect_dataset(combined_data)