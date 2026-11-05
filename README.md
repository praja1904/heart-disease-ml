# Heart Disease ML — Deployable Pipeline

Proyek machine learning untuk memprediksi risiko penyakit jantung menggunakan dataset UCI Heart Disease. Alur lengkap dari pembersihan data, training model, hingga deployment API.

## Struktur
- `src/data_prep.py`   → Tahap 1–6 (preprocessing)
- `src/train.py`       → Tahap 7–9 (split, scale, train, evaluate, export)
- `src/predict.py`     → Tahap 10 (inferensi)
- `api/app.py`         → REST API Flask

## Cara Pakai

1. **Install dependencies** (gunakan Python 3.11):
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt