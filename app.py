from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "churn_model.pkl"

sys.path.insert(0, str(ROOT / "src"))
from model_utils import validate_required_columns

app = Flask(__name__)

INPUT_FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

MODEL = None

if MODEL_PATH.exists():
    MODEL = joblib.load(MODEL_PATH)


def load_model():
    global MODEL

    if MODEL is None and MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)

    return MODEL


def validate_payload(payload):
    if not isinstance(payload, dict):
        return ["Request body must be a JSON object."]

    errors = []

    # Reuse the common required-column validation from model_utils.
    payload_df = pd.DataFrame([payload])
    missing = validate_required_columns(payload_df, include_target=False)

    if missing:
        errors.append(
            f"Missing required fields: {', '.join(missing)}"
        )

    for field in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if field in payload:
            try:
                float(payload[field])
            except (TypeError, ValueError):
                errors.append(f"{field} must be numeric.")

    if "SeniorCitizen" in payload:
        try:
            if int(float(payload["SeniorCitizen"])) not in (0, 1):
                errors.append("SeniorCitizen must be 0 or 1.")
        except (TypeError, ValueError):
            pass

    return errors


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": load_model() is not None
    })


@app.post("/predict")
def predict():
    model = load_model()

    if model is None:
        return jsonify({
            "error": "Model not found. Run the notebook first to train and save it."
        }), 503

    payload = request.get_json(silent=True)
    errors = validate_payload(payload)

    if errors:
        return jsonify({
            "error": "Invalid input",
            "details": errors
        }), 400

    row = {k: payload[k] for k in INPUT_FIELDS}
    df = pd.DataFrame([row])

    try:
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]

        classes = list(model.named_steps["classifier"].classes_)
        yes_index = classes.index("Yes") if "Yes" in classes else 1

        churn_probability = float(probabilities[yes_index])

        return jsonify({
            "prediction": str(prediction),
            "churn_probability": round(churn_probability, 4)
        })

    except Exception as exc:
        return jsonify({
            "error": "Prediction failed",
            "details": str(exc)
        }), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)