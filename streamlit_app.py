import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# Load trained model & artifacts
# =========================
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/artifacts/scaler.pkl"
ENCODERS_PATH = "models/artifacts/label_encoders.pkl"
FEATURES_PATH = "models/artifacts/feature_names.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_names = joblib.load(FEATURES_PATH)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="💼", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict the likelihood of churn.")

# =========================
# Input Form
# =========================
with st.form("input_form"):
    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1500.0)

    submitted = st.form_submit_button("🔮 Predict Churn")

# =========================
# Prediction Logic
# =========================
if submitted:
    # Create input DataFrame
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical columns
    for col, le in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col])
            except Exception:
                # handle unseen values
                input_df[col] = le.transform([le.classes_[0]])

    # Align columns with training features
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # fill missing engineered columns

    input_df = input_df[feature_names]

    # Scale numeric features
    X_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    pred = "Yes" if prob > 0.5 else "No"

    # Display
    st.markdown("---")
    st.subheader(f"Prediction: **{pred}**")
    st.metric("Churn Probability", f"{prob*100:.2f}%")

    if pred == "Yes":
        st.warning("⚠️ High risk of churn. Consider offering retention benefits.")
    else:
        st.success("✅ Low churn risk. Customer is likely to stay.")
