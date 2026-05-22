# EPS Churn Detection Pipeline

## Project Overview

This project converts the original Jupyter Notebook workflow into a clean, production-style machine learning pipeline for customer churn detection.

The model predicts whether a bank customer will exit/churn based on demographic, account, and activity features such as credit score, geography, gender, age, balance, number of products, credit card ownership, active membership, and estimated salary.

The original notebook logic was preserved as much as possible:
- Load `Churn_Modelling.csv`
- Drop identifier columns: `RowNumber`, `CustomerId`, `Surname`
- Encode `Gender`
- One-hot encode `Geography`
- Split the dataset using stratified train/test split
- Apply `RandomOverSampler` to the training set
- Standardize features using `StandardScaler`
- Train and evaluate the same model families:
  - KNN
  - Naive Bayes
  - SVM
  - Decision Tree
  - Random Forest
  - AdaBoost
  - Gradient Boosting
  - XGBoost

A small correction was made in the modular evaluation code: `classification_report` now uses the correct argument order, `y_true` followed by `y_pred`.

---

## Dataset Description

### Main dataset

File:

```text
data/raw/Churn_Modelling.csv
```

The churn dataset contains 10,000 customer records and the following columns:

| Column | Description |
|---|---|
| RowNumber | Row index identifier |
| CustomerId | Unique customer identifier |
| Surname | Customer surname |
| CreditScore | Customer credit score |
| Geography | Customer country: France, Germany, or Spain |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Number of years the customer has stayed |
| Balance | Customer account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Estimated customer salary |
| Exited | Target label: 1 = churned/exited, 0 = not churned |

### Extra uploaded dataset

File:

```text
data/raw/emails.csv
```

This file was included in the raw data folder because it was uploaded with the project files. It is not used in the churn notebook workflow.

---

## Project Structure

```text
eps_churn_detection_pipeline/
│
├── data/
│   ├── raw/
│   │   ├── Churn_Modelling.csv
│   │   └── emails.csv
│   └── processed/
│       └── features.csv
│
├── notebooks/
│   └── eps_churn_detection_original.ipynb
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── models.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── models/
│   └── best_model.joblib
│
├── outputs/
│   ├── figures/
│   └── reports/
│       ├── class_distribution.csv
│       ├── classification_reports.txt
│       ├── dataset_summary.json
│       └── evaluation_results.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation Steps

Open Command Prompt in the project folder.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

On Windows Command Prompt:

```bash
.venv\Scripts\activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Training Pipeline

From the project root folder, run:

```bash
python -m src.train
```

This command will:

1. Load the raw churn dataset.
2. Apply preprocessing and feature engineering.
3. Split the data into train and test sets.
4. Oversample the training data.
5. Scale the features.
6. Train all classifiers.
7. Evaluate all classifiers.
8. Save the best model artifact.

---

## How to Run the Streamlit App

After training the model, run:

```bash
streamlit run src/app.py
```

The app will open in your browser. If it does not open automatically, Command Prompt will show a local URL similar to:

```text
http://localhost:8501
```

Copy this URL and paste it into your browser.

---

## Expected Outputs

After running the training pipeline, the following outputs are generated:

### Model artifact

```text
models/best_model.joblib
```

This file contains:
- Best trained model
- Fitted scaler
- Feature column order
- Best model name
- Best model metrics

### Evaluation metrics

```text
outputs/reports/evaluation_results.csv
```

This file compares all trained models using:
- Train accuracy
- Test accuracy
- Train precision macro
- Test precision macro
- Train recall macro
- Test recall macro
- Train F1 macro
- Test F1 macro
- Test confusion matrix

### Classification reports

```text
outputs/reports/classification_reports.txt
```

This file contains detailed train and test classification reports for each model.

### Dataset summary

```text
outputs/reports/dataset_summary.json
```

This file contains:
- Number of rows
- Number of columns
- Missing values per column
- Duplicate row count

### Processed features

```text
data/processed/features.csv
```

This file stores the encoded feature table used by the pipeline.

---

## Command Summary

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
streamlit run src/app.py
```

---

## Notes

- The project is designed to run from the command line.
- The Streamlit app is located at `src/app.py`, as requested.
- The main notebook workflow was converted into modular Python scripts.
- The raw notebook is preserved in the `notebooks/` folder.
- The extra `emails.csv` file is stored in `data/raw/` but is not used by the churn prediction pipeline.
