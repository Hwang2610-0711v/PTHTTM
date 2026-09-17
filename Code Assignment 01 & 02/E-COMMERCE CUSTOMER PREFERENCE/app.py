import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
MODEL_FILE = "best_ecommerce_model.pkl"

# Nạp toàn bộ pipeline hoàn chỉnh
model_pipeline = joblib.load(MODEL_FILE)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ecommerce/v1/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    
    try:
        # Lấy dữ liệu thô gửi lên từ Client
        age = float(data.get("Age", 30))
        rating = float(data.get("Rating", 5))
        pos_feedback = float(data.get("Positive_Feedback_Count", 0))
        div_name = str(data.get("Division_Name", "General"))
        dept_name = str(data.get("Department_Name", "Tops"))
        class_name = str(data.get("Class_Name", "Blouses"))
        review_text = str(data.get("Review_Text", ""))
        
        # Tính toán các đặc trưng kỹ thuật số từ văn bản
        review_length = len(review_text)
        word_count = len(review_text.split())
        
        # Tạo DataFrame đúng định dạng cột như lúc train
        input_df = pd.DataFrame([{
            'Age': age,
            'Rating': rating,
            'Positive Feedback Count': pos_feedback,
            'Division Name': div_name,
            'Department Name': dept_name,
            'Class Name': class_name,
            'Review Text': review_text,
            'Title': '',
            'Review_Length': review_length,
            'Word_Count': word_count
        }])
        
        pred = int(model_pipeline.predict(input_df)[0])
        probas = model_pipeline.predict_proba(input_df)[0]
        confidence = round(float(probas[pred]) * 100, 2)
        
        return jsonify({
            "prediction": pred,
            "label": "Khuyên dùng (Recommended)" if pred == 1 else "Không khuyên dùng (Not Recommended)",
            "confidence": f"{confidence:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)