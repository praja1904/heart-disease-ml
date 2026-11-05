import joblib
import pandas as pd
from . import config

_model = _scaler = _features = None

def _load():
    global _model, _scaler, _features
    if _model is None:
        if not (config.MODEL_PATH.exists() and config.SCALER_PATH.exists() and config.FEATURES_PATH.exists()):
            raise FileNotFoundError("Artefak model belum ada. Jalankan `python run_pipeline.py` dulu.")
        _model    = joblib.load(config.MODEL_PATH)
        _scaler   = joblib.load(config.SCALER_PATH)
        _features = joblib.load(config.FEATURES_PATH)
    return _model, _scaler, _features

def predict_one(record: dict) -> dict:
    model, scaler, features = _load()
    # Mengisi nilai 0 untuk fitur yang tidak dikirim oleh user
    row = {f: record.get(f, 0) for f in features}
    X = pd.DataFrame([row])[features]

    X_scaled = scaler.transform(X)
    pred  = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])

    return {
        "prediction": pred,
        "label": "SAKIT JANTUNG" if pred == 1 else "NORMAL",
        "probability_sick": round(proba, 4),
    }

if __name__ == "__main__":
    contoh = {"age": 55, "sex": 1, "trestbps": 140, "chol": 250, "fbs": 0, "thalch": 150, "exang": 1, "oldpeak": 2.0, "ca": 1}
    print(predict_one(contoh))