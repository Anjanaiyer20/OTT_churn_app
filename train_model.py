import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv('data/ott_churn_data.csv')

# Encode categorical features
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['Subscription_Type'] = LabelEncoder().fit_transform(df['Subscription_Type'])

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Handle class imbalance
churn_counts = df['Churn'].value_counts()
scale_pos_weight = churn_counts[0] / churn_counts[1]

# ✅ Train the model with class weight adjustment
model = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'results/xgb_churn_model.pkl')
joblib.dump(scaler, 'results/scaler.pkl')
