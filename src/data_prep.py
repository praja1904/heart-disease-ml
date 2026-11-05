import numpy as np
import pandas as pd
from . import config

def load_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[LOAD] shape={df.shape}")
    return df

def inspect_data(df: pd.DataFrame) -> None:
    print("\n[INFO] Tipe data:")
    print(df.dtypes)
    print("\n[INFO] Distribusi target (num):")
    print(df["num"].value_counts().sort_index())

def validate_data(df: pd.DataFrame) -> None:
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    n_inf = np.isinf(df[num_cols].sum()).sum()
    print(f"[VALIDASI] Nilai inf: {n_inf}")
    print("[VALIDASI] Null per kolom:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

def build_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = (df["num"] > 0).astype(int)
    print(f"[LABEL] distribusi:\n{df['target'].value_counts()}")
    return df

def _to_bool_int(series: pd.Series) -> pd.Series:
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "male": 1, "female": 0}
    s = series.astype("string").str.strip().str.lower()
    return s.map(mapping)

def _safe_fill(series: pd.Series, fallback) -> pd.Series:
    if not series.isnull().any():
        return series
    mode_series = series.mode(dropna=True)
    fill_val = mode_series.iloc[0] if len(mode_series) > 0 else fallback
    return series.fillna(fill_val)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex"]   = _to_bool_int(df["sex"])
    df["fbs"]   = _to_bool_int(df["fbs"])
    df["exang"] = _to_bool_int(df["exang"])

    for col in ["cp", "restecg", "slope", "thal"]:
        df[col] = df[col].astype("string").str.strip().str.lower()

    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    for c in ["trestbps", "chol", "thalch", "oldpeak", "ca"]:
        if df[c].isnull().any():
            med = df[c].median()
            if pd.isna(med): med = 0.0
            df[c] = df[c].fillna(med)

    df["fbs"]   = _safe_fill(df["fbs"],   0)
    df["exang"] = _safe_fill(df["exang"], 0)
    df["sex"]   = _safe_fill(df["sex"],   0)
    df["restecg"] = _safe_fill(df["restecg"], "normal")
    df["slope"]   = _safe_fill(df["slope"],   "flat")
    df["thal"]    = _safe_fill(df["thal"],    "normal")
    df["cp"]      = _safe_fill(df["cp"],      "asymptomatic")

    df["sex"]   = df["sex"].astype(int)
    df["fbs"]   = df["fbs"].astype(int)
    df["exang"] = df["exang"].astype(int)
    for c in num_cols: df[c] = df[c].astype(float)

    df = df.drop(columns=["id", "dataset", "num"])
    n_dup = df.duplicated().sum()
    print(f"[CLEAN] duplikat dihapus: {n_dup}")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[CLEAN] shape setelah dibersihkan: {df.shape}")
    return df

def construct_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_enc = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal"], drop_first=True, dtype=int)
    print(f"[ENCODE] shape setelah encoding: {df_enc.shape}")

    X = df_enc.drop(columns=["target"])
    y = df_enc["target"]

    korelasi = X.corrwith(y).abs().sort_values(ascending=False)
    print("\n[KORELASI] fitur terhadap target:")
    print(korelasi)

    fitur_penting = korelasi[korelasi >= config.CORR_THRESHOLD].index.tolist()
    print(f"\n[SELEKSI] {len(fitur_penting)} fitur terpilih: {fitur_penting}")
    return X[fitur_penting], y

def prepare_pipeline() -> tuple[pd.DataFrame, pd.Series]:
    df = load_data()
    inspect_data(df)
    validate_data(df)
    df = build_label(df)
    df = clean_data(df)
    X, y = construct_features(df)
    return X, y