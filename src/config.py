"""Konfigurasi global path & parameter pipeline."""
from pathlib import Path

ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
MODEL_DIR      = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==== NAMA FILE DISERAGAMAN ====
RAW_DATA_PATH  = DATA_DIR / "heart_disease_uci.csv"
MODEL_PATH     = MODEL_DIR / "model_heart.pkl"       # <-- Diseragamkan
SCALER_PATH    = MODEL_DIR / "scaler_heart.pkl"
FEATURES_PATH  = MODEL_DIR / "features_heart.pkl"

# ==== Parameter training ====
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
CORR_THRESHOLD = 0.20  # Sesuai PDF