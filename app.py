
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load saved model bundle
MODEL_PATH = Path(__file__).parent / "models" / "churn_model.pkl"

@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle

bundle = load_model()
model = bundle["model"]
preprocessor = bundle["preprocessor"]

# Application title
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer "
    "is likely to leave the service."
)

st.divider()

st.subheader("Enter Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen", [0, 1]
        )

        partner = st.selectbox(
            "Partner", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents", ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)", 0, 100, 12
        )

        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.70
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=151.65
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    try:
        processed_data = preprocessor.transform(customer)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer May Leave!")
        else:
            st.success("✅ Customer Likely to Stay!")

        st.metric(
            "Estimated Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        st.info(
            "This is a machine learning estimate, "
            "not a guarantee of customer behaviour."
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.divider()
st.caption("Customer Churn Prediction | Machine Learning Capstone Project")
