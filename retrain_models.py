import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================================================
# CONFIG
# ==========================================================

DATA_DIR = "data/audio"
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")
MODEL_DIR = "models"
SAMPLE_RATE = 22050

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)

    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([
        mfcc, chroma, spectral_contrast,
        rolloff, centroid, zcr, rms
    ])

# ==========================================================
# LOAD DATASET
# ==========================================================

X = []
y = []

def load_folder(folder, label):
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    print(f"Found {len(files)} files in {folder}")

    for file in tqdm(files):
        path = os.path.join(folder, file)
        try:
            features = extract_features(path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"⚠ Skipping {file}: {e}")

load_folder(REAL_DIR, 0)
load_folder(FAKE_DIR, 1)

if len(X) == 0:
    raise RuntimeError("❌ No audio files processed. Check dataset.")

X = np.array(X)
y = np.array(y)

# ==========================================================
# SCALING + COVARIANCE
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled, rowvar=False)

# ==========================================================
# TRAIN MODELS
# ==========================================================

print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss"
)
xgb_model.fit(X_scaled, y)

print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    n_jobs=-1
)
rf_model.fit(X_scaled, y)

# ==========================================================
# SAVE ARTIFACTS
# ==========================================================

joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(cov_matrix, os.path.join(MODEL_DIR, "cov_matrix.pkl"))

print("\n✅ Retraining complete.")
print("Saved:")
print(" - xgb_model.pkl")
print(" - rf_model.pkl")
print(" - scaler.pkl")
print(" - cov_matrix.pkl")