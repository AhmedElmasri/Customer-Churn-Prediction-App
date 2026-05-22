"""Streamlit web application for customer churn prediction.

Run from the project root:

    streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make package imports work when running: streamlit run src/app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_PATH, MODEL_ARTIFACT_PATH, RAW_DATA_PATH
from src.predict import load_model_artifact, predict_customer


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Prediction App")
st.write(
    "This app predicts whether a bank customer is likely to exit/churn using "
    "the trained machine learning pipeline."
)


@st.cache_resource
def cached_artifact():
    """Load the trained model artifact once."""
    return load_model_artifact(MODEL_ARTIFACT_PATH)


with st.sidebar:
    st.header("Customer Input")

    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650, step=1)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", min_value=18, max_value=100, value=40, step=1)
    tenure = st.slider("Tenure", min_value=0, max_value=10, value=5, step=1)
    balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)

customer = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": estimated_salary,
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Summary")
    st.dataframe(pd.DataFrame([customer]), use_container_width=True)

    if st.button("Predict Churn", type="primary"):
        try:
            artifact = cached_artifact()
            result = predict_customer(customer, artifact)

            st.subheader("Prediction Result")
            st.success(f"Prediction: {result['label']}")
            st.write(f"Model used: **{result['model_name']}**")

            if result["churn_probability"] is not None:
                st.metric("Estimated Churn Probability", f"{result['churn_probability'] * 100:.2f}%")
            else:
                st.info("This model does not provide probability scores.")

        except FileNotFoundError:
            st.error(
                "The trained model file was not found. "
                "Open Command Prompt in the project folder and run: `python -m src.train`"
            )

with col2:
    st.subheader("Project Outputs")

    if MODEL_ARTIFACT_PATH.exists():
        st.success(f"Model artifact found: `{MODEL_ARTIFACT_PATH.name}`")
    else:
        st.warning("Model artifact is not available yet.")

    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        st.write("Model comparison metrics:")
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("Metrics will appear after running the training pipeline.")

st.divider()

with st.expander("View raw dataset sample"):
    if RAW_DATA_PATH.exists():
        sample_df = pd.read_csv(RAW_DATA_PATH).head(10)
        st.dataframe(sample_df, use_container_width=True)
    else:
        st.warning("Raw dataset file was not found in data/raw/.")
