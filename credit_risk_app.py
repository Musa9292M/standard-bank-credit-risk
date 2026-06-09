import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ====================== PASSWORD PROTECTION ======================
PASSWORD = "StandardBank2026"   # ← Change this to your preferred password

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login Required")
    password = st.text_input("Enter Password", type="password")
    if st.button("Login"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# ====================== MAIN APP ======================
st.set_page_config(page_title="Standard Bank Credit Risk", layout="wide")

# Professional Header
st.markdown("""
    <h1 style='text-align: center; color: #1E3A8A;'>
        🏦 Standard Bank Group
    </h1>
    <h2 style='text-align: center;'>Credit Risk Predictor</h2>
    <p style='text-align: center; color: #666;'>Powered by XGBoost + SHAP Explainability</p>
""", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Customer", "📊 Batch Scoring", "📖 User Guide"])

with tab1:
    st.sidebar.header("Customer Details")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 75, 28)
        duration = st.slider("Loan Duration (months)", 6, 72, 48)
        amount = st.number_input("Loan Amount", 500, 25000, 8500)
        installment_rate = st.slider("Installment Rate (1-4)", 1, 4, 4)
        existing_credits = st.slider("Existing Credits", 1, 4, 3)
    
    with col2:
        status = st.selectbox("Checking Account Status", ["< 0 DM", "0 - 200 DM", "> 200 DM", "No account"], index=0)
        credit_history = st.selectbox("Credit History", ["No credits taken", "All credits paid back", "Existing credits paid back", "Delay in paying in past", "Critical account"], index=4)
        purpose = st.selectbox("Purpose of Loan", ["Car", "Radio/TV", "Furniture", "Education", "Business", "Repairs", "Others"], index=3)

    if st.button("🚀 Calculate PD + SHAP Explanation", type="primary"):
        with st.spinner("Analyzing customer..."):
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

            # Text Explanation
            st.subheader("📝 Risk Explanation")
            if prob > 0.5:
                st.write("**Main drivers:** Long duration, large amount, poor checking account, and critical credit history.")
            elif prob > 0.25:
                st.write("**Moderate risk** due to combination of factors.")
            else:
                st.write("**Low risk profile** - Good credit behaviour.")

            # SHAP Bar Chart
            st.subheader("🔍 SHAP Feature Contributions")
            preprocessor = model.named_steps['preprocessor']
            X_processed = preprocessor.transform(input_data)
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
            shap_values = explainer.shap_values(X_processed)

            fig, ax = plt.subplots(figsize=(12, 7))
            shap.summary_plot(shap_values, X_processed, 
                            feature_names=preprocessor.get_feature_names_out(), 
                            plot_type="bar", show=False)
            plt.title("SHAP Bar Chart - Red = Increases Risk | Blue = Decreases Risk")
            st.pyplot(fig)

with tab2:
    st.header("Batch Scoring")
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        if st.button("Score All Customers"):
            with st.spinner("Scoring batch..."):
                predictions = model.predict_proba(df)[:, 1]
                df['PD_Score'] = predictions.round(4)
                df['Risk_Level'] = pd.cut(predictions, bins=[0, 0.25, 0.5, 1], labels=['Low', 'Medium', 'High'])
                st.success(f"Scored {len(df)} customers")
                st.dataframe(df)
                st.download_button("Download Results", df.to_csv(index=False), "batch_results.csv", "text/csv")

with tab3:
    st.header("📖 How to Use This App")
    st.markdown("""
    - **Single Customer**: Fill details → Click button → See PD + SHAP explanation
    - **Batch Scoring**: Upload Excel/CSV → Score multiple customers at once
    - **Red Bars** in SHAP = Features that **increase** default risk
    - **Blue Bars** = Features that **decrease** default risk
    """)

st.caption("Standard Bank Group | Final XGBoost Model with SHAP | Deployed on Streamlit Cloud")