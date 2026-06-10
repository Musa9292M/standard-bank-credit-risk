import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# ====================== PASSWORD ======================
PASSWORD = "StandardBank2026"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Standard Bank Credit Risk Predictor")
    st.markdown("### Enter Password to Access")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if pw == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# ====================== MAIN APP ======================
st.set_page_config(page_title="Standard Bank Credit Risk", layout="wide")

st.title("🏦 Standard Bank Credit Risk Predictor")
st.markdown("**XGBoost Model with Risk Scoring**")

@st.cache_resource
def load_model():
    return joblib.load("final_credit_risk_model.pkl")

model = load_model()

tab1, tab2 = st.tabs(["🔍 Single Customer", "📊 Batch Scoring"])

with tab1:
    st.sidebar.header("Customer Details")
    
    age = st.sidebar.slider("Age", 18, 75, 35)
    duration = st.sidebar.slider("Loan Duration (months)", 6, 72, 36)
    amount = st.sidebar.number_input("Loan Amount (R)", 1000, 50000, 12000)
    installment_rate = st.sidebar.slider("Installment Rate", 1, 4, 3)
    existing_credits = st.sidebar.slider("Existing Credits", 1, 4, 2)

    status = st.sidebar.selectbox("Checking Account", ["< 0 DM", "0 - 200 DM", "> 200 DM", "No account"])
    credit_history = st.sidebar.selectbox("Credit History", ["Critical account", "Delay in paying in past", "Existing credits paid back", "All credits paid back", "No credits taken"])
    purpose = st.sidebar.selectbox("Purpose", ["Business", "Education", "Car", "Radio/TV", "Furniture", "Repairs", "Others"])

    if st.button("🚀 Calculate PD", type="primary"):
        with st.spinner("Calculating..."):
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
            st.info("Prediction based on real benchmark data")

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
                st.download_button("Download Results", df.to_csv(index=False), "results.csv", "text/csv")

st.caption("Standard Bank Group | Risk Model App | Musa Vilakazi")