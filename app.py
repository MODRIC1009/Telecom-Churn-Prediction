from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
model = joblib.load("outputs/models/churn_model.pkl")

# Recreate feature structure
df = pd.read_csv("data/telco_churn.csv")
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("Churn_Yes", axis=1)
model_columns = X.columns


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Numeric inputs
    tenure = float(request.form["tenure"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])

    # Categorical inputs
    contract = request.form["Contract"]
    internet = request.form["InternetService"]
    security = request.form["OnlineSecurity"]
    support = request.form["TechSupport"]
    payment = request.form["PaymentMethod"]
    senior = int(request.form["SeniorCitizen"])
    dependents = request.form["Dependents"]

    # Create empty input frame
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    # Assign numeric values
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total
    input_data["SeniorCitizen"] = senior

    # Helper function for one-hot columns
    def set_if_exists(col):
        if col in input_data.columns:
            input_data[col] = 1

    # Set categorical values
    set_if_exists(f"Contract_{contract}")
    set_if_exists(f"InternetService_{internet}")
    set_if_exists(f"OnlineSecurity_{security}")
    set_if_exists(f"TechSupport_{support}")
    set_if_exists(f"PaymentMethod_{payment}")
    set_if_exists(f"Dependents_{dependents}")

    # Prediction
    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.3:
        result = f"Low churn risk (Probability: {prob:.2f})"
    elif prob < 0.6:
        result = f"Medium churn risk (Probability: {prob:.2f})"
    else:
        result = f"High churn risk (Probability: {prob:.2f})"

    return render_template("index.html", prediction_text=result)



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
