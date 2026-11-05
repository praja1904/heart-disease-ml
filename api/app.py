from flask import Flask, request, jsonify
from src.predict import predict_one

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Heart Disease Prediction API", "endpoints": {"GET /health": "Cek status", "POST /predict": "Prediksi"}})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body harus object"}), 400
        result = predict_one(payload)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)