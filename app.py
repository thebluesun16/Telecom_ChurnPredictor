import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# ------------------------------
# Load data and model
# ------------------------------
df = pd.read_csv("tel_churn.csv")   # Already encoded and tenure_grouped
model = pickle.load(open("model.sav", "rb"))  # XGBoost model

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction using XGBoost")
st.write("Enter customer details below to predict whether they are likely to churn.")

st.markdown("---")

# ------------------------------
# Input Fields
# ------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["0", "1"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure_group = st.selectbox("Tenure Group", [
        "1 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"
    ])

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges (₹)", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input("Total Charges (₹)", min_value=0.0, value=2000.0)

# ------------------------------
# Prediction Logic
# ------------------------------
if st.button("🔍 Predict Churn"):
    # Create DataFrame for user input
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [int(SeniorCitizen)],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'tenure_group': [tenure_group]
    })

    # One-hot encode & align with training columns
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=df.drop('Churn', axis=1).columns, fill_value=0)

    # Predict with XGBoost model
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)[:, 1][0] * 100

    # ------------------------------
    # Display results
    # ------------------------------
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"🚨 The customer is **likely to churn.**")
        st.metric(label="Confidence Level", value=f"{probability:.2f}%")
    else:
        st.success(f"✅ The customer is **likely to stay.**")
        st.metric(label="Confidence Level", value=f"{probability:.2f}%")

st.markdown("---")
