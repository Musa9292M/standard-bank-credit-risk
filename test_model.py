import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import xgboost as xgb

np.random.seed(42)

print("🚀 Final Stable Improved Model\n")

# Load German Credit Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = ['status','duration','credit_history','purpose','amount','savings','employment',
                'installment_rate','personal_status','other_debtors','residence','property','age',
                'other_installment','housing','existing_credits','job','dependents','telephone',
                'foreign_worker','default']

df = pd.read_csv(url, sep=' ', header=None, names=column_names)
df['default'] = (df['default'] == 2).astype(int)

categorical_features = ['status', 'credit_history', 'purpose', 'savings', 'employment', 
                       'personal_status', 'other_debtors', 'property', 'housing', 'job', 'foreign_worker']
numerical_features = ['duration', 'amount', 'age', 'installment_rate', 'existing_credits']

X = df[categorical_features + numerical_features]
y = df['default']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"✅ Final XGBoost ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, pipeline.predict(X_test)))

# Feature Importance
importances = pd.DataFrame({
    'Feature': pipeline.named_steps['preprocessor'].get_feature_names_out(),
    'Importance': pipeline.named_steps['classifier'].feature_importances_
}).sort_values(by='Importance', ascending=False).head(15)

print("\n🔍 Top Feature Importance:")
print(importances)

# Save
joblib.dump(pipeline, 'final_credit_risk_model.pkl')
print("\n💾 Final model saved as 'final_credit_risk_model.pkl'")

# Export Report
test_results = pd.DataFrame(X_test)
test_results['actual_default'] = y_test.values
test_results['pd_score'] = y_pred_proba
test_results.to_excel('final_german_credit_report.xlsx', index=False)
print("📊 Report exported to 'final_german_credit_report.xlsx'")