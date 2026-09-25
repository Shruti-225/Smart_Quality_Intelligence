from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_FEATURE_COUNT = 30


def load_processed_data(
    file_path: Path
) -> pd.DataFrame:
    """Load the cleaned production dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)


def prepare_features_and_target(
    data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate process features from the quality target."""

    excluded_columns = [
        "quality_label",
        "timestamp"
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    X = data[feature_columns]
    y = data["quality_label"]

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:
    """Split data into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def calculate_mutual_information(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> pd.Series:
    """Calculate mutual information between features and target."""

    scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=RANDOM_STATE
    )

    return pd.Series(
        scores,
        index=X_train.columns,
        name="mutual_information"
    )


def calculate_random_forest_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> pd.Series:
    """Calculate feature importance using Random Forest."""

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return pd.Series(
        model.feature_importances_,
        index=X_train.columns,
        name="random_forest_importance"
    )


def create_feature_ranking(
    mutual_information: pd.Series,
    random_forest_importance: pd.Series
) -> pd.DataFrame:
    """Combine feature-selection scores into one ranking."""

    ranking = pd.concat(
        [
            mutual_information,
            random_forest_importance
        ],
        axis=1
    )

    ranking["mi_rank"] = (
        ranking["mutual_information"]
        .rank(ascending=False)
    )

    ranking["rf_rank"] = (
        ranking["random_forest_importance"]
        .rank(ascending=False)
    )

    ranking["combined_rank"] = (
        ranking["mi_rank"] +
        ranking["rf_rank"]
    )

    ranking = ranking.sort_values(
        "combined_rank"
    )

    return ranking


def select_top_features(
    ranking: pd.DataFrame,
    number_of_features: int
) -> list[str]:
    """Return the highest-ranked features."""

    return ranking.head(
        number_of_features
    ).index.tolist()


def save_feature_ranking(
    ranking: pd.DataFrame,
    output_path: Path
) -> None:
    """Save feature ranking to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ranking.to_csv(
        output_path
    )


def save_selected_features(
    selected_features: list[str],
    output_path: Path
) -> None:
    """Save selected feature names."""

    pd.DataFrame({
        "feature": selected_features
    }).to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    processed_file = (
        project_root
        / "data"
        / "processed"
        / "production_cleaned.csv"
    )

    reports_dir = (
        project_root
        / "outputs"
        / "reports"
    )

    data = load_processed_data(
        processed_file
    )

    X, y = prepare_features_and_target(
        data
    )

    X_train, X_test, y_train, y_test = (
        split_dataset(X, y)
    )

    print("\n========== DATA SPLIT ==========")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("\nTraining quality distribution:")
    print(y_train.value_counts())

    print("\nTesting quality distribution:")
    print(y_test.value_counts())

    print("\n========== MUTUAL INFORMATION ==========")

    mutual_information = (
        calculate_mutual_information(
            X_train,
            y_train
        )
    )

    print(
        mutual_information
        .sort_values(ascending=False)
        .head(10)
    )

    print(
        "\n========== RANDOM FOREST IMPORTANCE =========="
    )

    random_forest_importance = (
        calculate_random_forest_importance(
            X_train,
            y_train
        )
    )

    print(
        random_forest_importance
        .sort_values(ascending=False)
        .head(10)
    )

    ranking = create_feature_ranking(
        mutual_information,
        random_forest_importance
    )

    selected_features = select_top_features(
        ranking,
        TOP_FEATURE_COUNT
    )

    save_feature_ranking(
        ranking,
        reports_dir / "feature_importance.csv"
    )

    save_selected_features(
        selected_features,
        reports_dir / "selected_features.csv"
    )

    print(
        "\n========== SELECTED FEATURES =========="
    )

    for feature in selected_features:
        print(feature)

    print(
        "\nFeature-selection reports saved to:"
    )

    print(reports_dir)