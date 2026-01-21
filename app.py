import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Credit Card Fraud Detection")
st.write("Upload transaction data CSV or enter transaction details manually.")

# Upload CSV
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Remove 'Class' column if present
    if "Class" in data.columns:
        data_features = data.drop("Class", axis=1)
    else:
        data_features = data

    # Scale features
    try:
        data_scaled = scaler.transform(data_features)
    except ValueError as e:
        st.error(f"Feature mismatch: {e}")
    else:
        # Predict
        predictions = model.predict(data_scaled)
        probabilities = model.predict_proba(data_scaled)[:,1]

        results = data_features.copy()
        results["Fraud Probability"] = np.round(probabilities, 4)
        results["Prediction"] = predictions
        st.subheader("Prediction Results")
        st.dataframe(results)

else:
    st.write("Or enter transaction details manually below (fill all features).")
    
    # Create manual input fields dynamically based on training features
    feature_names = [col for col in scaler.feature_names_in_ if col != "Class"]
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0,1]
        st.write(f"Prediction: {pred} (Fraud Probability: {prob:.4f})")
