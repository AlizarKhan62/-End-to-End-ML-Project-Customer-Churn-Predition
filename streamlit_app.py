import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "artifacts", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "artifacts", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "artifacts", "label_encoders.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "models", "artifacts", "feature_names.pkl")

# =========================
# Load artifacts
# =========================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="Customer Churn Prediction", page_icon="💼", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f8fafc, #e0f2fe);
        color: #1f2937;
    }
    .header-title {
        font-size: 2.5rem;
        color: #1e3a8a;
        font-weight: bold;
        text-align: center;
    }
    .header-subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #475569;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        font-weight: bold;
    }
    .logo-img {
      display: block;
      margin-left: auto;
      margin-right: auto;
      border-radius: 50px;
    }

    </style>
""", unsafe_allow_html=True)

# =========================
# Header + Brand Logo
# =========================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    # Placeholder for logo
    st.image("icons/Telecom_arg_logo.png", width=100)  # 👈 You can replace this path with your actual logo later
with col_title:
    st.markdown("<div class='header-title'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>Predict the likelihood of a customer leaving your service</div>", unsafe_allow_html=True)

st.markdown("---")


# =========================
# Sidebar Form
# =========================
with st.sidebar.form("input_form"):
    st.header("📝 Customer Information")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    st.markdown("---")
    st.subheader("📡 Service Details")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    
    st.markdown("---")
    st.subheader("💰 Charges")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1500.0)
    
    submitted = st.form_submit_button("🔮 Predict Churn")

# =========================
# Prediction Logic (unchanged)
# =========================
if submitted:
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
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col])
            except Exception:
                input_df[col] = le.transform([le.classes_[0]])

    # Add missing engineered columns if any
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[feature_names]

    # Scale numeric features
    X_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    pred = "Yes" if prob > 0.5 else "No"

    # =========================
    # Display results
    # =========================
    st.markdown("---")
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.subheader(f"Prediction: **{pred}**")
    with col2:
        st.metric("Churn Probability", f"{prob*100:.2f}%")
    
    if pred == "Yes":
        st.warning("⚠️ High risk of churn. Consider offering retention benefits.")
    else:
        st.success("✅ Low churn risk. Customer is likely to stay.")
