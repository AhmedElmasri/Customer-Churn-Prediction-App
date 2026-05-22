"""Project configuration.

All paths are defined relative to the project root so the code can run
from any machine after extracting the project folder.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "Churn_Modelling.csv"
EXTRA_RAW_EMAIL_DATA_PATH = RAW_DATA_DIR / "emails.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

MODEL_ARTIFACT_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = REPORTS_DIR / "evaluation_results.csv"
CLASSIFICATION_REPORTS_PATH = REPORTS_DIR / "classification_reports.txt"
PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"

TARGET_COLUMN = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Numerical columns expected by the original churn dataset.
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

# Columns produced by the original notebook logic:
# - Gender is label encoded with Female dropped, so only Gender_Male remains.
# - Geography is one-hot encoded without dropping any country column.
CATEGORICAL_FEATURES = ["Gender", "Geography"]
