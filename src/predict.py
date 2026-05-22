"""Prediction utilities for trained churn models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_ARTIFACT_PATH
from src.preprocessing import align_features, encode_features


def load_model_artifact(model_path: str | Path = MODEL_ARTIFACT_PATH) -> dict:
    """Load the trained model, scaler, and metadata."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact was not found at {model_path}. "
            "Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def predict_customer(customer_data: dict[str, Any], artifact: dict | None = None) -> dict:
    """Predict churn for a single customer input dictionary."""
    if artifact is None:
        artifact = load_model_artifact()

    raw_input = pd.DataFrame([customer_data])
    X = encode_features(raw_input)
    X = align_features(X, artifact["feature_columns"])
    X_scaled = artifact["scaler"].transform(X)

    model = artifact["model"]
    prediction = int(model.predict(X_scaled)[0])

    churn_probability = None
    if hasattr(model, "predict_proba"):
        churn_probability = float(model.predict_proba(X_scaled)[0][1])

    return {
        "prediction": prediction,
        "label": "Exited / Churned" if prediction == 1 else "Not Exited / Not Churned",
        "churn_probability": churn_probability,
        "model_name": artifact.get("model_name", "Unknown"),
    }


def predict_batch(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Predict churn for a dataframe of customers."""
    if artifact is None:
        artifact = load_model_artifact()

    X = encode_features(df)
    X = align_features(X, artifact["feature_columns"])
    X_scaled = artifact["scaler"].transform(X)

    model = artifact["model"]
    predictions = model.predict(X_scaled)

    output = df.copy()
    output["Predicted_Exited"] = predictions.astype(int)

    if hasattr(model, "predict_proba"):
        output["Churn_Probability"] = model.predict_proba(X_scaled)[:, 1]

    return output
