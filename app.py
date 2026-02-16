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
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("Churn_Yes", axis=1)
model_columns = X.columns


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    tenure = float(request.form["tenure"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])

    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total

    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.3:
        result = f"Low churn risk (Probability: {prob:.2f})"
    elif prob < 0.6:
        result = f"Medium churn risk (Probability: {prob:.2f})"
    else:
        result = f"High churn risk (Probability: {prob:.2f})"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
