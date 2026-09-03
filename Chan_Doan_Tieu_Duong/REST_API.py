import pickle
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_FILE = "diabetes.sav"
model = pickle.load(open(MODEL_FILE, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/diabetes/v1/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    # The deployed model in this project was trained with the selected
    # features in this order: Glucose, BMI, Age.
    try:
        glucose = float(data["Glucose"])
        bmi = float(data["BMI"])
        age = float(data["Age"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Glucose, BMI and Age must be provided as numbers."
        }), 400

    features = [[glucose, bmi, age]]
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = round(float(np.max(probabilities)) * 100, 2)

    return jsonify({
        "prediction": prediction,
        "confidence": f"{confidence:.2f}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    app.run(host="0.0.0.0", port=5000)
    app.run(host="0.0.0.0", port=5000)
