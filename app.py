"""
app.py
------
This is the main Streamlit application for the Multi-Disease Prediction System.
It integrates trained machine learning models for Heart Disease, Diabetes,
and Breast Cancer, providing an interactive user interface for prediction.

All models and preprocessing objects are loaded from the saved artifacts.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os


# =========================
# APPLICATION CONFIGURATION
# =========================
# Set Streamlit page settings
st.set_page_config(page_title="Multi Disease Prediction System", layout="wide")
st.title("🩺 Multi-Disease Prediction System")

# Directory where trained models are stored
MODEL_DIR = "models"


# =========================
# LOAD TRAINED MODELS
# =========================
# Each disease has multiple algorithms and a corresponding scaler
MODELS = {
    "Heart": {
        "lr": joblib.load(f"{MODEL_DIR}/heart_lr.pkl"),
        "rf": joblib.load(f"{MODEL_DIR}/heart_rf.pkl"),
        "xgb": joblib.load(f"{MODEL_DIR}/heart_xgb.pkl"),
        "svm": joblib.load(f"{MODEL_DIR}/heart_svm.pkl"),
        "scaler": joblib.load(f"{MODEL_DIR}/heart_scaler.pkl"),
    },
    "Diabetes": {
        "lr": joblib.load(f"{MODEL_DIR}/diabetes_lr.pkl"),
        "rf": joblib.load(f"{MODEL_DIR}/diabetes_rf.pkl"),
        "xgb": joblib.load(f"{MODEL_DIR}/diabetes_xgb.pkl"),
        "svm": joblib.load(f"{MODEL_DIR}/diabetes_svm.pkl"),
        "scaler": joblib.load(f"{MODEL_DIR}/diabetes_scaler.pkl"),
    },
    "Cancer": {
        "lr": joblib.load(f"{MODEL_DIR}/cancer_lr.pkl"),
        "rf": joblib.load(f"{MODEL_DIR}/cancer_rf.pkl"),
        "xgb": joblib.load(f"{MODEL_DIR}/cancer_xgb.pkl"),
        "svm": joblib.load(f"{MODEL_DIR}/cancer_svm.pkl"),
        "scaler": joblib.load(f"{MODEL_DIR}/cancer_scaler.pkl"),
    }
}


# =========================
# FEATURE ORDER (LOCKED)
# =========================
# Feature order must strictly match the order used during model training

HEART_FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal",
    "bp_category","chol_category"
]

DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BMI",
    "Insulin",
    "Age"
]

CANCER_FEATURES = [
    "mean radius","mean texture","mean perimeter",
    "mean area","mean smoothness"
]

# Mapping diseases to their corresponding feature sets
FEATURES = {
    "Heart": HEART_FEATURES,
    "Diabetes": DIABETES_FEATURES,
    "Cancer": CANCER_FEATURES
}


# =========================
# SIDEBAR CONTROLS
# =========================
# User selects disease type and machine learning algorithm
disease = st.sidebar.selectbox("Select Disease", ["Heart", "Diabetes", "Cancer"])
algo = st.sidebar.selectbox("Select Algorithm", ["lr", "rf", "xgb", "svm"])


# =========================
# USER INPUT INTERFACE
# =========================
st.subheader(f"🧾 Enter {disease} Patient Data")

# Dictionary to store user inputs
inputs = {}

# Dynamically generate numeric inputs based on selected disease
for f in FEATURES[disease]:
    inputs[f] = st.number_input(f.replace("_"," ").title(), value=0.0)


# =========================
# PREDICTION LOGIC
# =========================
if st.button("🔍 Predict"):

    # Load corresponding scaler and model
    scaler = MODELS[disease]["scaler"]
    model = MODELS[disease][algo]

    # Create NumPy array in the same feature order as training
    X = np.array([[inputs[f] for f in FEATURES[disease]]])

    # Apply scaling and generate prediction
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    # Interpret prediction result
    if disease == "Cancer":
        result = "Malignant (Cancer)" if pred == 1 else "Benign"
    else:
        result = "Disease Detected" if pred == 1 else "No Disease Detected"

    # Display result to the user
    st.success(f"✅ Prediction Result: **{result}**")
