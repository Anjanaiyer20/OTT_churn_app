import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('results/xgb_churn_model.pkl')
scaler = joblib.load('results/scaler.pkl')

st.set_page_config(page_title="OTT Churn Prediction", layout="centered")
st.title("📺 OTT Customer Churn Prediction")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 80, 30)
subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
watch_time = st.slider("Avg Daily Watch Time (hrs)", 0, 10, 3)
devices = st.slider("Number of Devices Used", 1, 10, 2)
last_login = st.slider("Days Since Last Login", 0, 60, 7)
support_calls = st.slider("Customer Support Calls", 0, 10, 1)

# Encode input
gender_encoded = 1 if gender == 'Male' else 0
subscription_map = {"Basic": 0, "Standard": 1, "Premium": 2}
subscription_encoded = subscription_map[subscription]

input_data = np.array([[gender_encoded, age, subscription_encoded, watch_time,
                        devices, last_login, support_calls]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Churn"):
    pred = model.predict(input_scaled)[0]
    if pred == 1:
        st.error("⚠️ The customer is likely to **CHURN**.")
    else:
        st.success("✅ The customer is likely to **STAY**.")
