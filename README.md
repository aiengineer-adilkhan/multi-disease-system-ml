# 🩺 Multi-Disease Prediction System (Machine Learning)

**Author:** Adil Khan
**Repository:** `multi-disease-system-ml`

---

## 📌 Project Overview

The **Multi-Disease Prediction System** is a Machine Learning–based application designed to predict the presence of multiple diseases using patient medical data. This project currently supports prediction for:

* ❤️ Heart Disease
* 🩸 Diabetes
* 🎗️ Cancer

The system integrates **multiple ML algorithms**, performs proper data preprocessing and scaling, and provides a **Streamlit-based interactive web interface** for real-time predictions. The project is developed following professional standards and is suitable for academic submission and internship evaluation.

---

## 🎯 Objectives

* Build accurate ML models for disease prediction
* Compare multiple algorithms on the same dataset
* Ensure consistent preprocessing using scalers
* Deploy a unified application using Streamlit
* Maintain clean, modular, and reusable code

---

## 🧠 Machine Learning Algorithms Used

For each disease, the following algorithms are implemented:

* **Logistic Regression (LR)**
* **Support Vector Machine (SVM)**
* **Random Forest (RF)**
* **XGBoost (XGB)**

Each model is trained independently and saved for reuse in the main application.

---

## 📂 Project Structure

```
Multi-Disease-System-ML/
│
├── app.py                  # Main Streamlit application
├── .gitignore              # Files/folders excluded from GitHub
│
├── data/                   # Datasets
│   ├── heart.csv
│   ├── diabetes.csv
│   └── cancer.csv
│
├── training/               # Model training scripts (run once)
│   ├── heart.py
│   ├── diabetes.py
│   └── cancer.py
│
├── models/                 # Saved trained models & scalers
│   ├── *_lr.pkl
│   ├── *_rf.pkl
│   ├── *_svm.pkl
│   ├── *_xgb.pkl
│   └── *_scaler.pkl
│
└── venv/                   # Virtual environment (ignored in GitHub)
```

---

## 🧾 Features Used

### ❤️ Heart Disease

* Age, Sex, Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Rest ECG
* Max Heart Rate
* Exercise Angina
* Oldpeak, Slope, CA, Thal
* BP Category, Cholesterol Category

### 🩸 Diabetes

* Pregnancies
* Glucose
* BMI
* Insulin
* Age

### 🎗️ Cancer

* Mean Radius
* Mean Texture
* Mean Perimeter
* Mean Area
* Mean Smoothness

---

## 🖥️ Application Interface

The **Streamlit web application** allows users to:

* Select disease type
* Choose ML algorithm
* Enter patient medical data
* Get instant prediction results

The UI is simple, clean, and user-friendly.

---

## ▶️ How to Run the Project

### 1️⃣ Activate Virtual Environment

```powershell
venv\Scripts\Activate.ps1
```

### 2️⃣ Run the Streamlit App

```powershell
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📊 Model Evaluation Metrics

Models were evaluated using standard ML metrics:

* **Accuracy** – Overall correctness
* **Recall** – Ability to detect positive cases
* **F1-Score** – Balance between precision and recall

These metrics ensure reliability, especially in medical prediction scenarios.

---

## 🔐 Notes

* Training scripts are designed to be run **once** only
* Saved models are reused in `app.py`
* Feature order is strictly maintained to ensure correct predictions
* `.gitignore` prevents unnecessary files from being uploaded

---

## 🚀 Future Improvements

* Add more diseases
* Improve UI/UX
* Deploy on Streamlit Cloud or Heroku
* Add database support
* Implement explainable AI (XAI)

---

## 🏁 Conclusion

This project demonstrates the complete Machine Learning pipeline—from data preprocessing and model training to deployment. It reflects practical ML skills, clean coding practices, and real-world application development.

---

### 📬 Contact

**Adil Khan**
GitHub: [https://github.com/aiengineer-adilkhan](https://github.com/aiengineer-adilkhan)

---

⭐ *If you like this project, feel free to star the repository!*
