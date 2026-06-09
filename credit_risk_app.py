import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Standard Bank Credit Risk", layout="wide")

st.title("🏦 Standard Bank Credit Risk Predictor")
st.markdown("**Final XGBoost Model with SHAP Explainability**")

# Password protection
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pw = st.text_input("Enter Password", type="password")
    if st.button("Login"):
        if pw == "StandardBank2026":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    return joblib.load('final_credit_risk_model.pkl')

model = load_model()

# Single Customer Tab
st.header("Single Customer Scoring")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 75, 28)
    duration = st.slider("Loan Duration (months)", 6, 72, 48)
    amount = st.number_input("Loan Amount", 500, 25000, 8500)
    installment_rate = st.slider("Installment Rate", 1, 4, 4)
    existing_credits = st.slider("Existing Credits", 1, 4, 3)

with col2:
    status = st.selectbox("Checking Account", ["< 0 DM", "0 - 200 DM", "> 200 DM", "No account"])
    credit_history = st.selectbox("Credit History", ["No credits taken", "All credits paid back", "Existing credits paid back", "Delay in paying in past", "Critical account"])
    purpose = st.selectbox("Purpose", ["Car", "Radio/TV", "Furniture", "Education", "Business", "Repairs", "Others"])

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

        st.info("**Note:** Red bars in SHAP = features increasing default risk")

st.caption("Deployed on Streamlit Cloud | Musa Vilakazi - Senior Risk Analytics Engineer")