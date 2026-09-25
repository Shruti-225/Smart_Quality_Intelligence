from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(
    data_path: Path,
    selected_features_path: Path
) -> tuple[pd.DataFrame, list[str]]:

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    if not selected_features_path.exists():
        raise FileNotFoundError(
            f"Selected features not found: {selected_features_path}"
        )

    data = pd.read_csv(data_path)

    selected_features = pd.read_csv(
        selected_features_path
    )["feature"].astype(str).tolist()

    return data, selected_features


def prepare_data(
    data: pd.DataFrame,
    selected_features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:

    X = data[selected_features]
    y = data["quality_label"]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> tuple[LogisticRegression, StandardScaler]:

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    return model, scaler


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    scaler: StandardScaler | None = None
) -> dict:

    if scaler is not None:
        X_test = scaler.transform(X_test)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"\n========== {model_name.upper()} =========="
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Good",
                "Defective"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


def save_production_predictions(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path
) -> None:

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Threshold selected from previous analysis
    threshold = 0.40

    predictions = (
        probabilities >= threshold
    ).astype(int)

    predictions_df = pd.DataFrame({
        "record_index": X_test.index,
        "actual_label": y_test.values,
        "production_probability": probabilities,
        "production_prediction": predictions
    })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nProduction predictions saved to:"
    )

    print(output_path)


def save_model_results(
    results: list[dict],
    output_path: Path
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(results).to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "data"
        / "processed"
        / "production_cleaned.csv"
    )

    selected_features_path = (
        project_root
        / "outputs"
        / "reports"
        / "selected_features.csv"
    )

    results_path = (
        project_root
        / "outputs"
        / "reports"
        / "model_comparison.csv"
    )

    production_predictions_path = (
        project_root
        / "outputs"
        / "reports"
        / "production_predictions.csv"
    )

    model_path = (
        project_root
        / "outputs"
        / "models"
        / "random_forest_model.pkl"
    )

    features_path = (
        project_root
        / "outputs"
        / "models"
        / "selected_features.pkl"
    )

    data, selected_features = load_data(
        data_path,
        selected_features_path
    )

    X, y = prepare_data(
        data,
        selected_features
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\n========== MODEL DATA ==========")

    print(
        f"Selected features: {len(selected_features)}"
    )

    print(
        f"Training records: {len(X_train)}"
    )

    print(
        f"Testing records: {len(X_test)}"
    )

    logistic_model, scaler = (
        train_logistic_regression(
            X_train,
            y_train
        )
    )

    random_forest_model = (
        train_random_forest(
            X_train,
            y_train
        )
    )

    results = []

    results.append(
        evaluate_model(
            logistic_model,
            X_test,
            y_test,
            "Logistic Regression",
            scaler
        )
    )

    results.append(
        evaluate_model(
            random_forest_model,
            X_test,
            y_test,
            "Random Forest"
        )
    )

    save_model_results(
        results,
        results_path
    )

    print(
        "\nModel comparison saved to:"
    )

    print(results_path)

    save_production_predictions(
        random_forest_model,
        X_test,
        y_test,
        production_predictions_path
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        random_forest_model,
        model_path
    )

    joblib.dump(
        selected_features,
        features_path
    )

    print("\nRandom Forest model saved to:")
    print(model_path)

    print("\nSelected features saved to:")
    print(features_path)