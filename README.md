About the Project

The Multi-Disease Prediction System is a machine learning–based healthcare application developed to predict the likelihood of multiple diseases using patient medical data. The core idea behind this project is to bring different disease prediction models into one unified, intelligent system that follows proper machine learning practices from start to deployment.

In this project, separate machine learning pipelines were designed for Heart Disease, Diabetes, and Cancer. For each disease, the dataset was carefully analyzed, cleaned, and preprocessed. Important medical features were selected, and numerical data was standardized using appropriate scaling techniques to ensure consistent and reliable predictions.

Multiple machine learning algorithms were trained independently for each disease, including Logistic Regression (LR), Support Vector Machine (SVM), Random Forest (RF), and XGBoost (XGB). Training these models separately allowed meaningful performance comparison and ensured that the most suitable algorithm could be selected during prediction time. Once trained, both the models and their corresponding scalers were saved and reused, avoiding unnecessary retraining and maintaining prediction accuracy.

To make the system practical and user-friendly, all trained models were integrated into a Streamlit-based web application. The application allows users to select a disease, choose a machine learning algorithm, input medical data, and receive instant prediction results. Special care was taken to maintain feature order consistency, which is critical in medical machine learning systems.

This project demonstrates the complete machine learning lifecycle, including data handling, feature engineering, model training, evaluation using metrics such as Accuracy, Recall, and F1-Score, and final deployment. It reflects real-world ML development standards and is designed to be easily extendable for future diseases, improved interfaces, and cloud deployment.

Overall, the Multi-Disease Prediction System showcases practical machine learning skills, clean code organization, and an end-to-end approach to building intelligent healthcare applications.
