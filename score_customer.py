import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the final model
model = joblib.load('final_credit_risk_model.pkl')

def score_customer(customer_data):
    """Score a single customer with better handling"""
    df = pd.DataFrame([customer_data])
    
    # Make prediction
    prob = model.predict_proba(df)[:, 1][0]
    risk = "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.25 else "LOW"
    
    print(f"Probability of Default: {prob:.2%} → {risk} Risk")
    return prob, risk

# ==================== Test Customers ====================

# Test 1: High Risk Example
high_risk = {
    'duration': 48,
    'amount': 8000,
    'age': 25,
    'installment_rate': 4,
    'existing_credits': 2,
    'status': '< 0 DM',
    'credit_history': 'Critical account',
    'purpose': 'Education',
    'savings': 'unknown',
    'employment': '<1 year',
    'personal_status': 'male single',
    'other_debtors': 'none',
    'property': 'car or other',
    'housing': 'rent',
    'job': 'unskilled',
    'foreign_worker': 'yes'
}

print("=== High Risk Test ===")
score_customer(high_risk)

# Test 2: Low Risk Example
low_risk = {
    'duration': 12,
    'amount': 1500,
    'age': 42,
    'installment_rate': 1,
    'existing_credits': 1,
    'status': '> 200 DM',
    'credit_history': 'All credits paid back',
    'purpose': 'Car',
    'savings': 'unknown',
    'employment': '>=7 years',
    'personal_status': 'male single',
    'other_debtors': 'none',
    'property': 'real estate',
    'housing': 'own',
    'job': 'skilled',
    'foreign_worker': 'yes'
}

print("\n=== Low Risk Test ===")
score_customer(low_risk)

print("\n✅ You can now add more test cases easily!")