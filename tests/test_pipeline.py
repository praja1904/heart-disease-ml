import os
from src import config
from src.predict import predict_one


def test_artifacts_exist():
    assert os.path.exists(config.MODEL_PATH)
    assert os.path.exists(config.SCALER_PATH)
    assert os.path.exists(config.FEATURES_PATH)


def test_predict_output_schema():
    rec = {"age": 55, "sex": 1, "trestbps": 140, "chol": 250,
           "fbs": 0, "thalch": 150, "exang": 1, "oldpeak": 2.0, "ca": 1}
    out = predict_one(rec)
    assert "prediction" in out
    assert "label" in out
    assert 0 <= out["probability_sick"] <= 1