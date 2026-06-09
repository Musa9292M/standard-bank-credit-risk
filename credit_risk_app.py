credit_risk_app.py
@@ -0,0 +1,102 @@
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Standard Bank Credit Risk", layout="wide")

st.title("🏦 Standard Bank Credit Risk Predictor")
st.markdown("**Final XGBoost Model with SHAP Explainability**")

@st.cache_resource
def load_model():
    return joblib.load('final_credit_risk_model.pkl')

model = load_model()

tab1, tab2 = st.tabs(["🔍 Single Customer", "📊 Batch Scoring"])
