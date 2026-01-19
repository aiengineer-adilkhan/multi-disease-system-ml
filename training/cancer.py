"""
cancer.py
---------
This script trains multiple machine learning models for Breast Cancer prediction.
It performs data loading, feature selection, scaling, model training, evaluation,
and saves trained models and preprocessing objects for application use.
"""

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score


# =========================
# PATHS (SAFE & ABSOLUTE)
# =========================
# Resolve base project directory to avoid path-related issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
# Load breast cancer dataset
df = pd.read_csv(DATA_PATH)


# =========================
# TARGET VARIABLE
# =========================
# Diagnosis column:
# M = Malignant (1)
# B = Benign (0)
TARGET_COL = "diagnosis"
df[TARGET_COL] = df[TARGET_COL].map({"M": 1, "B": 0})


# =========================
# SELECTED FEATURES (FIXED)
# =========================
# Selected based on medical relevance and model simplicity
FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean"
]

X = df[FEATURES]
y = df[TARGET_COL]


# =========================
# FEATURE SCALING
# =========================
# Scaling is required for models sensitive to feature magnitude
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for consistent preprocessing during inference
joblib.dump(scaler, os.path.join(MODEL_DIR, "cancer_scaler.pkl"))


# =========================
# TRAIN-TEST SPLIT
# =========================
# Stratification maintains class balance in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# MODEL DEFINITIONS
# =========================
# Multiple algorithms are trained to compare performance
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}


# =========================
# TRAIN, EVALUATE & SAVE
# =========================
for name, model in models.items():

    # Train model on training data
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate classification performance
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display evaluation metrics
    print("\n==============================")
    print(f"🔹 {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save models using filenames expected by app.py
    if name == "Logistic Regression":
        joblib.dump(model, os.path.join(MODEL_DIR, "cancer_lr.pkl"))
    elif name == "SVM":
        joblib.dump(model, os.path.join(MODEL_DIR, "cancer_svm.pkl"))
    elif name == "Random Forest":
        joblib.dump(model, os.path.join(MODEL_DIR, "cancer_rf.pkl"))
    elif name == "XGBoost":
        joblib.dump(model, os.path.join(MODEL_DIR, "cancer_xgb.pkl"))


print("\n✅ Cancer models trained, evaluated & safely overwritten")
