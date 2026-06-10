# 🏦 Standard Bank Credit Risk Predictor

A user-friendly web application built to predict the **Probability of Default (PD)** for credit customers using a robust XGBoost machine learning model with **SHAP explainability**.

---

## ✨ Features

- **Single Customer Scoring**: Instant PD prediction + detailed SHAP explanation (Red/Blue bars)
- **Batch Scoring**: Upload Excel or CSV files to score multiple customers at once
- **Risk Classification**: HIGH / MEDIUM / LOW risk levels with clear explanations
- **Password Protected**: Secure access
- **Professional UI**: Clean and easy to use

---

## 🚀 Live App

**Link**: [https://standard-bank-credit-risk-h3mzxh3fztjfdaue6cy33.streamlit.app](https://standard-bank-credit-risk-h3mzxh3fztjfdaue6cy33.streamlit.app)

**Password**: `StandardBank2026`

---

## 🛠️ Technologies Used

- **Model**: XGBoost (ROC AUC ≈ 0.77)
- **Explainability**: SHAP
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud
- **CI/CD**: GitHub Actions
- **Data**: German Credit Dataset

---

## 🔄 CI/CD with GitHub Actions

This project uses **GitHub Actions** for automated testing and quality checks.

- Every time code is pushed to the `main` branch, GitHub automatically:
  - Sets up the Python environment
  - Installs dependencies
  - Runs basic checks
- This ensures code quality and reliability with every update.

---

## 📁 Project Structure
