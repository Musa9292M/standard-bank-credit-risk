# 🏦 Standard Bank Credit Risk Predictor

A user-friendly web application built to predict the **Probability of Default (PD)** for credit customers using a robust XGBoost machine learning model with **SHAP explainability**.

---

## ✨ Features

- **Single Customer Scoring**: Get instant PD prediction + detailed SHAP explanation (Red/Blue bars)
- **Batch Scoring**: Upload Excel or CSV files to score multiple customers at once
- **Risk Classification**: HIGH / MEDIUM / LOW risk levels with clear explanations
- **Password Protected**: Secure access
- **Professional UI**: Clean and easy to use

---

## 🚀 How to Use the Live App

**Live Link**: [https://standard-bank-credit-risk-h3mzxh3fztjfdaue6cy33.streamlit.app](https://standard-bank-credit-risk-h3mzxh3fztjfdaue6cy33.streamlit.app)

**Password**: `StandardBank2026`

### Single Customer Tab
1. Enter customer details in the sidebar
2. Click **"Calculate PD + SHAP Explanation"**
3. View PD percentage, risk level, explanation, and SHAP chart

### Batch Scoring Tab
1. Upload an Excel or CSV file
2. Click **"Score All Customers"**
3. Download the results with PD scores and risk levels

---

## 🛠️ Technologies Used

- **Model**: XGBoost (ROC AUC ≈ 0.77)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud
- **Data**: German Credit Dataset (industry benchmark)

---

## 📁 Project Structure
