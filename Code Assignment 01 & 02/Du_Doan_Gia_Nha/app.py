import os
import json
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_FILE = "best_house_price_model.pkl"
METADATA_FILE = "house_model_metadata.json"

model = joblib.load(MODEL_FILE)
try:
    metadata = json.load(open(METADATA_FILE, encoding="utf-8"))
except FileNotFoundError:
    metadata = {
        "numeric_features": ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"],
        "categorical_features": ["House direction", "Balcony direction", "Legal status", "Furniture state", "Location"]
    }

NUMERIC = metadata["numeric_features"]
CATEGORICAL = metadata["categorical_features"]
FEATURES = NUMERIC + CATEGORICAL

@app.route("/")
def home():
    return render_template(
        "index.html",
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL
    )

@app.route("/house/v1/predict", methods=["POST"])
def predict_house():
    data = request.get_json(silent=True) or {}
    row = {}

    try:
        for col in NUMERIC:
            row[col] = float(data[col])
        for col in CATEGORICAL:
            row[col] = str(data.get(col, ""))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Missing or invalid house fields."}), 400

    import pandas as pd
    X_new = pd.DataFrame([row], columns=FEATURES)
    prediction = float(model.predict(X_new)[0])

    return jsonify({
        "prediction": round(prediction, 2)
    })

if __name__== "__main__":
    app.run(host="0.0.0.0", port=5000)
