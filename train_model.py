import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv('data/ott_churn_data.csv')

# Encode categorical
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['Subscription_Type'] = LabelEncoder().fit_transform(df['Subscription_Type'])

# Features and labels
X = df.drop('Churn', axis=1)
y = df['Churn']

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost
# New line (clean, no warning):
model = XGBClassifier(eval_metric='logloss')

model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'results/xgb_churn_model.pkl')
joblib.dump(scaler, 'results/scaler.pkl')
