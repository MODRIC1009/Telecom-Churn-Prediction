import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")

st.title("Telecom Customer Churn Prediction")

# Load model
model = joblib.load("outputs/models/churn_model.pkl")

# Load dataset to recreate feature structure
df = pd.read_csv("data/telco_churn.csv")
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("Churn_Yes", axis=1)
model_columns = X.columns

# Input fields
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=800.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
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

senior = st.selectbox("Senior Citizen", [0, 1])
dependents = st.selectbox("Dependents", ["Yes", "No"])

if st.button("Predict Churn"):
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total
    input_data["SeniorCitizen"] = senior

    def set_if_exists(col):
        if col in input_data.columns:
            input_data[col] = 1

    set_if_exists(f"Contract_{contract}")
    set_if_exists(f"InternetService_{internet}")
    set_if_exists(f"OnlineSecurity_{security}")
    set_if_exists(f"TechSupport_{support}")
    set_if_exists(f"PaymentMethod_{payment}")
    set_if_exists(f"Dependents_{dependents}")

    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.3:
        st.success(f"Low churn risk (Probability: {prob:.2f})")
    elif prob < 0.6:
        st.warning(f"Medium churn risk (Probability: {prob:.2f})")
    else:
        st.error(f"High churn risk (Probability: {prob:.2f})")
