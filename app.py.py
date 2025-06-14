
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved model and scaler
model = joblib.load("logistic_model_sample.joblib")
scaler = joblib.load("standard_scaler_sample.joblib")

st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter transaction details below to check if it's **fraudulent** or **genuine**.")

# Input fields for all 30 features
time = st.number_input("Time (in seconds)", min_value=0.0)
amount = st.number_input("Amount", min_value=0.0)

# 28 PCA-based features (V1–V28)
v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", format="%.4f")
    v_features.append(val)

if st.button("Predict"):
    # Create input array
    input_data = np.array([[time, *v_features, amount]])

    # Scale Time and Amount (first and last columns)
    input_df = pd.DataFrame(input_data, columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This transaction is predicted as **FRAUD** (Probability: {proba:.2f})")
    else:
        st.success(f"✅ This transaction is predicted as **GENUINE** (Probability of fraud: {proba:.2f})")
