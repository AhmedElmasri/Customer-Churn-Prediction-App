"""Data loading utilities for the churn project."""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH


def load_churn_data(data_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the customer churn dataset.

    Parameters
    ----------
    data_path:
        Path to the churn CSV file.

    Returns
    -------
    pd.DataFrame
        Raw churn dataframe.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Place Churn_Modelling.csv in data/raw/ or pass a valid path."
        )

    return pd.read_csv(data_path)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a compact dataset summary used by reports or debugging."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
    }
