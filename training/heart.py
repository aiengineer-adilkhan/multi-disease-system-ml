"""
heart.py
--------
This script trains multiple machine learning models for Heart Disease prediction.
It handles data loading, preprocessing, model training, evaluation, and saving
all required artifacts for later use in the main application.
"""

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score


# -----------------------------
# Load dataset
# -----------------------------
# Reading the heart disease dataset from the data directory
df = pd.read_csv("../data/heart.csv")


# -----------------------------
# Target column
# -----------------------------
# 'num' indicates heart disease presence:
# 0 = No disease
# >0 = Disease present
TARGET_COL = "num"

# Convert target into binary classification (0 or 1)
df[TARGET_COL] = df[TARGET_COL].apply(lambda x: 1 if x > 0 else 0)


# -----------------------------
# Encode categorical columns
# -----------------------------
# Identifying categorical and boolean columns
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns

# Dictionary to store encoders for reuse during prediction
encoders = {}

for col in categorical_cols:
    # Label Encoding converts categorical values into numeric form
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le


# -----------------------------
# Split features & target
# -----------------------------
# X contains all input features
# y contains the target variable
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# -----------------------------
# Handle missing values
# -----------------------------
# Median strategy is robust to outliers and suitable for medical data
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)


# -----------------------------
# Train-test split
# -----------------------------
# Stratification ensures equal class distribution in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -----------------------------
# Feature scaling
# -----------------------------
# Scaling improves performance for distance-based models like SVM and Logistic Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# Model definitions
# -----------------------------
# Multiple models are trained to compare performance
models = {
    "lr": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "svm": SVC(probability=True),
    "xgb": XGBClassifier(eval_metric="logloss")
}


# -----------------------------
# Model save directory
# -----------------------------
# All trained models and preprocessing objects are saved here
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Train, Evaluate & Save Models
# -----------------------------
for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Generate predictions on test data
    y_pred = model.predict(X_test)

    # Evaluate model performance
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display evaluation metrics
    print("\n==============================")
    print(f"🔹 Model: {name.upper()}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save trained model for deployment
    model_path = f"{MODEL_DIR}/heart_{name}.pkl"
    joblib.dump(model, model_path)


# -----------------------------
# Save preprocessing objects
# -----------------------------
# These are required during prediction to ensure consistency
joblib.dump(scaler, f"{MODEL_DIR}/heart_scaler.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/heart_encoders.pkl")
joblib.dump(imputer, f"{MODEL_DIR}/heart_imputer.pkl")


print("\n✅ Heart disease models trained, evaluated & safely saved")
