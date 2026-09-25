from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
ALERT_THRESHOLD = 0.10


def load_data(
    data_path: Path,
    selected_features_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Load processed production data and selected features."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    if not selected_features_path.exists():
        raise FileNotFoundError(
            f"Selected features file not found: "
            f"{selected_features_path}"
        )

    data = pd.read_csv(data_path)

    selected_features = (
        pd.read_csv(selected_features_path)
        ["feature"]
        .astype(str)
        .tolist()
    )

    return data, selected_features


def prepare_data(
    data: pd.DataFrame,
    selected_features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare model features and target."""

    X = data[selected_features]
    y = data["quality_label"]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series
):
    """Create a stratified train-test split."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """Train the Random Forest defect prediction model."""

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


def evaluate_thresholds(
    y_test: pd.Series,
    probabilities
) -> pd.DataFrame:
    """Evaluate model performance at different thresholds."""

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    return pd.DataFrame(results)


def save_threshold_results(
    results: pd.DataFrame,
    output_path: Path
) -> None:
    """Save threshold analysis results."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        output_path,
        index=False
    )


def plot_precision_recall(
    y_test: pd.Series,
    probabilities,
    output_path: Path
) -> None:
    """Create the precision-recall curve."""

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            probabilities
        )
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Precision-Recall Curve"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def plot_confusion_matrix(
    y_test: pd.Series,
    probabilities,
    threshold: float,
    output_path: Path
) -> None:
    """Create a confusion matrix using the selected threshold."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Good",
            "Defective"
        ]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - Threshold {threshold:.2f}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def plot_roc_curve(
    y_test: pd.Series,
    probabilities,
    output_path: Path
) -> None:
    """Create the ROC curve."""

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_test,
            probabilities
        )
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Random Forest (AUC = {auc_score:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - Defect Prediction"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def save_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_path: Path
) -> None:
    """Save Random Forest feature importance rankings."""

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        output_path,
        index=False
    )


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_path: Path
) -> None:
    """Plot the top 15 important features."""

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .head(15)
        .sort_values(
            "importance"
        )
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        importance["feature"].astype(str),
        importance["importance"]
    )

    plt.xlabel(
        "Random Forest Importance"
    )

    plt.ylabel(
        "Process Feature"
    )

    plt.title(
        "Top Process Features for Defect Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


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

    reports_dir = (
        project_root
        / "outputs"
        / "reports"
    )

    figures_dir = (
        project_root
        / "outputs"
        / "figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    data, selected_features = load_data(
        data_path,
        selected_features_path
    )

    X, y = prepare_data(
        data,
        selected_features
    )

    X_train, X_test, y_train, y_test = (
        split_data(
            X,
            y
        )
    )

    print("\n========== MODEL DATA ==========")

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    print(
        f"Training records: "
        f"{len(X_train)}"
    )

    print(
        f"Testing records: "
        f"{len(X_test)}"
    )

    model = train_model(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        "\n========== MODEL ROC-AUC =========="
    )

    print(
        f"ROC-AUC: {auc_score:.4f}"
    )

    threshold_results = evaluate_thresholds(
        y_test,
        probabilities
    )

    print(
        "\n========== THRESHOLD ANALYSIS =========="
    )

    print(
        threshold_results.to_string(
            index=False
        )
    )

    threshold_file = (
        reports_dir
        / "threshold_analysis.csv"
    )

    save_threshold_results(
        threshold_results,
        threshold_file
    )

    plot_precision_recall(
        y_test,
        probabilities,
        figures_dir
        / "precision_recall_curve.png"
    )

    plot_confusion_matrix(
        y_test,
        probabilities,
        ALERT_THRESHOLD,
        figures_dir
        / "confusion_matrix.png"
    )

    plot_roc_curve(
        y_test,
        probabilities,
        figures_dir
        / "roc_curve.png"
    )

    save_feature_importance(
        model,
        selected_features,
        reports_dir
        / "model_feature_importance.csv"
    )

    plot_feature_importance(
        model,
        selected_features,
        figures_dir
        / "feature_importance.png"
    )

    print(
        "\n========== OUTPUTS GENERATED =========="
    )

    print(
        f"Threshold report:\n{threshold_file}"
    )

    print(
        "\nFigures saved to:"
    )

    print(
        figures_dir
    )