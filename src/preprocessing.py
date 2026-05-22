"""Preprocessing and feature engineering utilities.

This module follows the same main feature logic used in the notebook:
1. Drop identifier columns: RowNumber, CustomerId, Surname.
2. Convert Gender into Gender_Male.
3. One-hot encode Geography.
4. Split into features X and target y.
5. Oversample the training set.
6. Standardize features.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DROP_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready feature columns from raw churn rows.

    The function is intentionally robust for both full datasets and single-row
    inference inputs. It avoids the common single-row `get_dummies(drop_first=True)`
    problem where a Male row could lose the Gender_Male column.
    """
    data = df.copy()

    # Remove columns that are identifiers and not useful for prediction.
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns], errors="ignore")

    # Remove target if present because this function returns only features.
    if TARGET_COLUMN in data.columns:
        data = data.drop(columns=[TARGET_COLUMN])

    # Validate required base columns before feature engineering.
    required = set(NUMERIC_FEATURES + ["Gender", "Geography"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    # Keep the numeric columns in the same order used by the notebook.
    features = data[NUMERIC_FEATURES].copy()

    # Notebook logic: pd.get_dummies(Gender, drop_first=True) gives Gender_Male.
    gender = data["Gender"].astype(str).str.strip().str.lower()
    features["Gender_Male"] = (gender == "male").astype(int)

    # Notebook logic: pd.get_dummies(Geography, prefix="Geography") with no drop.
    geography_dummies = pd.get_dummies(data["Geography"], prefix="Geography").astype(int)
    features = pd.concat([features, geography_dummies], axis=1)

    return features


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X and y from the raw dataframe."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    X = encode_features(df)
    y = df[TARGET_COLUMN].copy()
    return X, y


def align_features(X: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Align inference features to the exact columns learned during training."""
    aligned = X.copy()

    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0

    # Drop any extra columns and enforce training order.
    aligned = aligned[list(feature_columns)]
    return aligned


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the same stratified train/test split used in the notebook."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def oversample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply RandomOverSampler to the training data only."""
    # Imported inside the function so the Streamlit inference app can run
    # even before imbalanced-learn is installed. Training still requires it.
    from imblearn.over_sampling import RandomOverSampler

    oversampler = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[object, object, StandardScaler]:
    """Fit StandardScaler on training data and transform train/test data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
