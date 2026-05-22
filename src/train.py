"""Train and evaluate the churn detection models.

Run from the project root:

    python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    CLASSIFICATION_REPORTS_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_FEATURES_PATH,
    RAW_DATA_PATH,
    REPORTS_DIR,
)
from src.data_loader import get_dataset_summary, load_churn_data
from src.evaluate import evaluate_classifier
from src.models import get_models
from src.preprocessing import (
    oversample_training_data,
    prepare_features_and_target,
    scale_features,
    split_dataset,
)


def ensure_directories() -> None:
    """Create required output directories if they do not exist."""
    for path in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_basic_eda(df: pd.DataFrame) -> None:
    """Save lightweight EDA outputs for documentation and inspection."""
    ensure_directories()

    summary_path = REPORTS_DIR / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(get_dataset_summary(df), indent=4),
        encoding="utf-8",
    )

    class_distribution = (
        df["Exited"]
        .value_counts()
        .rename_axis("Exited")
        .reset_index(name="count")
        .sort_values("Exited")
    )
    class_distribution.to_csv(REPORTS_DIR / "class_distribution.csv", index=False)


def train_all_models(data_path: str | Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Run the full training pipeline and save the best model artifact."""
    ensure_directories()

    df = load_churn_data(data_path)
    save_basic_eda(df)

    X, y = prepare_features_and_target(df)
    processed = X.copy()
    processed["Exited"] = y
    processed.to_csv(PROCESSED_FEATURES_PATH, index=False)

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train_resampled, y_train_resampled = oversample_training_data(X_train, y_train)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_resampled, X_test)

    model_results = []
    report_blocks = []
    fitted_models = {}

    for model_name, model in get_models().items():
        print(f"Training {model_name}...")
        model.fit(X_train_scaled, y_train_resampled)
        fitted_models[model_name] = model

        scores = evaluate_classifier(
            model,
            X_train_scaled,
            y_train_resampled,
            X_test_scaled,
            y_test,
        )

        result_row = {
            "model": model_name,
            "train_accuracy": scores["train_accuracy"],
            "test_accuracy": scores["test_accuracy"],
            "train_precision_macro": scores["train_precision_macro"],
            "test_precision_macro": scores["test_precision_macro"],
            "train_recall_macro": scores["train_recall_macro"],
            "test_recall_macro": scores["test_recall_macro"],
            "train_f1_macro": scores["train_f1_macro"],
            "test_f1_macro": scores["test_f1_macro"],
            "test_confusion_matrix": scores["test_confusion_matrix"],
        }
        model_results.append(result_row)

        report_blocks.append(
            f"\n{'=' * 80}\n"
            f"MODEL: {model_name}\n"
            f"{'=' * 80}\n"
            f"TRAIN REPORT\n{scores['train_report']}\n"
            f"TEST REPORT\n{scores['test_report']}\n"
            f"TEST CONFUSION MATRIX: {scores['test_confusion_matrix']}\n"
        )

    metrics_df = pd.DataFrame(model_results).sort_values(
        by=["test_f1_macro", "test_accuracy"],
        ascending=False,
    )
    metrics_df.to_csv(METRICS_PATH, index=False)
    CLASSIFICATION_REPORTS_PATH.write_text("\n".join(report_blocks), encoding="utf-8")

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "scaler": scaler,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics_df.iloc[0].to_dict(),
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(f"Saved model artifact: {MODEL_ARTIFACT_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

    return metrics_df, artifact


if __name__ == "__main__":
    train_all_models()
