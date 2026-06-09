import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ====================== PASSWORD PROTECTION ======================
PASSWORD = "StandardBank2026"   # Change this to your desired password

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Standard Bank Credit Risk Predictor")
    st.markdown("### Login Required")
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

# Professional Header with Logo
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #1E3A8A; margin-bottom: 0;">🏦 Standard Bank Group</h1>
        <h2 style="margin-top: 0; color: #1E40AF;">Credit Risk Predictor</h2>
        <p style="color: #666;">Powered by XGBoost + SHAP Explainability</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Customer Scoring", "📊 Batch Scoring", "📖 User Guide"])

with tab1:
    st.sidebar.header("📋 Customer Details")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 75, 28)
        duration = st.slider("Loan Duration (months)", 6, 72, 48)
        amount = st.number_input("Loan Amount", 500, 25000, 8500)
        installment_rate = st.slider("Installment Rate (1=low → 4=high)", 1, 4, 3)
        existing_credits = st.slider("Existing Credits", 1, 4, 2)
    
    with col2:
        status = st.selectbox("Checking Account Status", ["< 0 DM", "0 - 200 DM", "> 200 DM", "No account"], index=0)
        credit_history = st.selectbox("Credit History", ["No credits taken", "All credits paid back", "Existing credits paid back", "Delay in paying in past", "Critical account"], index=4)
        purpose = st.selectbox("Purpose of Loan", ["Car", "Radio/TV", "Furniture", "Education", "Business", "Repairs", "Others"], index=0)

    if st.button("🚀 Calculate PD + SHAP Explanation", type="primary"):
        with st.spinner("Analyzing customer risk..."):
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

            st.success(f"**Predicted Probability of Default: {prob:.2%}**")

            if risk == "HIGH":
                st.error("🔴 HIGH RISK - Strong likelihood of default")
            elif risk == "MEDIUM":
                st.warning("🟡 MEDIUM RISK - Additional review recommended")
            else:
                st.success("🟢 LOW RISK - Generally acceptable")

            st.progress(float(prob))

            # Text Explanation
            st.subheader("📝 Risk Explanation")
            if prob > 0.5:
                st.write("**Main drivers of high risk:** Long loan duration, large amount, poor checking account status, and critical credit history.")
            elif prob > 0.25:
                st.write("**Moderate risk factors present.**")
            else:
                st.write("**Low risk profile** - Good credit history and reasonable loan terms.")

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

            st.info("**Red bars** push the prediction toward **default (higher risk)**\n**Blue bars** push toward **no default (lower risk)**")

with tab2:
    st.header("Batch Scoring")
    uploaded_file = st.file_uploader("Upload Excel or CSV file with customer data", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
        if st.button("Score All Customers"):
            with st.spinner("Scoring batch..."):
                predictions = model.predict_proba(df)[:, 1]
                df['PD_Score'] = predictions.round(4)
                df['Risk_Level'] = pd.cut(predictions, bins=[0, 0.25, 0.5, 1], labels=['Low', 'Medium', 'High'])
                st.success(f"Successfully scored {len(df)} customers")
                st.dataframe(df)
                st.download_button("📥 Download Results", df.to_csv(index=False), "batch_risk_results.csv", "text/csv")

with tab3:
    st.header("📖 User Guide")
    st.markdown("""
    **How to use this app:**

    - **Single Customer**: Fill in the details → Click the button → View PD + SHAP explanation
    - **Batch Scoring**: Upload an Excel or CSV file → Score multiple customers at once
    - **Red Bars** in SHAP chart = Features that **increase** default risk
    - **Blue Bars** = Features that **decrease** default risk

    **Password**: `StandardBank2026` (change in code if needed)
    """)

st.caption("Standard Bank Group | Senior Risk Analytics Engineer | Musa Vilakazi")