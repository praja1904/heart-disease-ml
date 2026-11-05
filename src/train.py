import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .data_prep import prepare_pipeline

def train() -> None:
    X, y = prepare_pipeline()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print("\nData berhasil dibagi menjadi set Training dan Testing.")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Training
    model = LogisticRegression(
        class_weight='balanced', 
        solver='liblinear', 
        random_state=config.RANDOM_STATE, 
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    print("Pelatihan model Regresi Logistik selesai. Model siap dievaluasi.")

    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    print("\n### Classification Report ###")
    print(classification_report(y_test, y_pred, target_names=['NORMAL (0)', 'SAKIT (1)']))

    # Visualisasi
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL (0)', 'SAKIT (1)'], 
                yticklabels=['NORMAL (0)', 'SAKIT (1)'])
    plt.title('Confusion Matrix - Heart Disease')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    plt.savefig(config.MODEL_DIR / "confusion_matrix.png")
    plt.close() # Mencegah plot menggantung di terminal

    # Export
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(list(X.columns), config.FEATURES_PATH)
    print(f"\n[EXPORT] Model  -> {config.MODEL_PATH}")
    print(f"[EXPORT] Scaler -> {config.SCALER_PATH}")
    print(f"[EXPORT] Feats  -> {config.FEATURES_PATH}")

if __name__ == "__main__":
    train()