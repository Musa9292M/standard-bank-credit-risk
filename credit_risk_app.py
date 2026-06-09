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

model = load_model()   # ← This fixes all "model is not defined" errors

tab1, tab2 = st.tabs(["🔍 Single Customer", "📊 Batch Scoring"])

with tab1:
    st.sidebar.header("Customer Details")
    
    age = st.sidebar.slider("Age", 18, 75, 28)
    duration = st.sidebar.slider("Loan Duration (months)", 6, 72, 48)
    amount = st.sidebar.number_input("Loan Amount", 500, 25000, 8500)
    installment_rate = st.sidebar.slider("Installment Rate", 1, 4, 3)
    existing_credits = st.sidebar.slider("Existing Credits", 1, 4, 2)

    status = st.sidebar.selectbox("Checking Account", ["< 0 DM", "0 - 200 DM", "> 200 DM", "No account"])
    credit_history = st.sidebar.selectbox("Credit History", ["No credits taken", "All credits paid back", "Existing credits paid back", "Delay in paying in past", "Critical account"])
    purpose = st.sidebar.selectbox("Purpose", ["Car", "Radio/TV", "Furniture", "Education", "Business", "Repairs", "Others"])

    if st.button("🚀 Calculate PD + Explanation", type="primary"):
        with st.spinner("Analyzing..."):
            input_data = pd.DataFrame([{
                'duration': duration, 'amount': amount, 'age': age,
                'installment_rate': installment_rate, 'existing_credits': existing_credits,
                'status': status, 'credit_history': credit_history, 'purpose': purpose,
                'savings': 'unknown', 'employment': '1-4 years', 'personal_status': 'male single',
                'other_debtors': 'none', 'property': 'real estate', 'housing': 'own',
                'job': 'skilled', 'foreign_worker': 'yes'
            }])

            prob = model.predict_proba(input_data)[:, 1][0]
            risk = "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.25 else "LOW"

            st.success(f"**Predicted PD: {prob:.2%}**")

            if risk == "HIGH":
                st.error("🔴 HIGH RISK")
            elif risk == "MEDIUM":
                st.warning("🟡 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")

            st.progress(float(prob))

            st.info("**Risk Explanation:** Long duration, large amount, and poor credit status are main contributors.")

with tab2:
    st.header("Batch Scoring")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        if st.button("Score All Customers"):
            with st.spinner("Scoring..."):
                predictions = model.predict_proba(df)[:, 1]
                df['PD_Score'] = predictions.round(4)
                df['Risk_Level'] = pd.cut(predictions, bins=[0, 0.25, 0.5, 1], labels=['Low', 'Medium', 'High'])
                st.success(f"Scored {len(df)} customers")
                st.dataframe(df)
                st.download_button("Download Results", df.to_csv(index=False), "batch_results.csv", "text/csv")

st.caption("Fixed Version | Stable Model Loading")