import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set Streamlit page configuration
st.set_page_config(page_title="OTT Churn Predictor", page_icon="🎥", layout="centered")
st.title("OTT Customer Churn Prediction 📊")
st.markdown("""
This app predicts whether a customer is likely to **churn** or **stay** based on their OTT service usage.
""")

# Load and preprocess dataset
df = pd.read_csv("ott_churn_data.csv")

# Encode categorical features
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_sub = LabelEncoder()
df['Subscription_Type'] = le_sub.fit_transform(df['Subscription_Type'])

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance
churn_counts = y.value_counts()
scale_pos_weight = churn_counts[0] / churn_counts[1]

model = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, use_label_encoder=False)
model.fit(X_train, y_train)

# Sidebar for input features
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", le_gender.classes_)
age = st.sidebar.slider("Age", 18, 60, 30)
sub_type = st.sidebar.selectbox("Subscription Type", le_sub.classes_)
watch_time = st.sidebar.slider("Daily Watch Time (hrs)", 0.0, 6.0, 2.0, step=0.1)
num_devices = st.sidebar.slider("Number of Devices", 1, 5, 2)
last_login = st.sidebar.slider("Days Since Last Login", 0, 90, 15)
support_calls = st.sidebar.slider("Customer Support Calls", 0, 10, 2)

# Predict button
if st.sidebar.button("Predict Churn"):
    input_df = pd.DataFrame({
        'Gender': [le_gender.transform([gender])[0]],
        'Age': [age],
        'Subscription_Type': [le_sub.transform([sub_type])[0]],
        'Watch_Time': [watch_time],
        'Num_Devices': [num_devices],
        'Last_Login_Days': [last_login],
        'Customer_Support_Calls': [support_calls]
    })

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    # Debug output
    st.write("Prediction (0=Stay, 1=Churn):", prediction[0])
    st.write("Churn Probability:", round(prob, 2))

    if prediction[0] == 1:
        st.error(f"⚠️ The customer is likely to CHURN! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ The customer is likely to STAY. (Probability of churn: {prob:.2f})")


# Optional: Show evaluation metrics
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    st.write("### Accuracy:", round(acc, 2))
    st.write("### Confusion Matrix:")
    st.dataframe(cm)
    st.write("### Classification Report:")
    st.text(cr)

# Footer
st.markdown("""
---
Built with ❤️ using Streamlit and XGBoost.
""")
