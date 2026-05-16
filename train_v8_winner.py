import os
import random
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# =========================
# CONFIG
# =========================
SEED = 42
SAMPLE_RATE = 22050
N_SPLITS = 5

random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "audio")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORIES = ["real", "fake"]

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, spectral_contrast, tonnetz])

# =========================
# LOAD DATA
# =========================
print("🔍 Loading dataset...")

X = []
y = []

for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_PATH, category)

    for file in os.listdir(folder):
        if not file.lower().endswith((".wav", ".mp3", ".flac")):
            continue

        file_path = os.path.join(folder, file)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)

print(f"✅ Total samples: {len(X)}")

# =========================
# HANDLE CLASS IMBALANCE
# =========================
scale_pos_weight = (len(y) - sum(y)) / sum(y)

# =========================
# CROSS VALIDATION
# =========================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

accuracies = []
roc_aucs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

    print(f"\n📊 Fold {fold}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    base_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        reg_lambda=2,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        use_label_encoder=False
    )

    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)

    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    accuracies.append(acc)
    roc_aucs.append(roc_auc)

    print("Accuracy:", round(acc * 100, 2), "%")
    print("ROC-AUC:", round(roc_auc, 4))

print("\n🏆 FINAL RESULTS")
print("Mean Accuracy:", round(np.mean(accuracies) * 100, 2), "%")
print("Mean ROC-AUC:", round(np.mean(roc_aucs), 4))

# =========================
# FINAL TRAIN ON FULL DATA
# =========================
print("\n🚀 Training final deployable model...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

base_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_lambda=2,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=SEED,
    use_label_encoder=False
)

final_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
final_model.fit(X_scaled, y)

# =========================
# SAVE MODEL
# =========================
joblib.dump(final_model, os.path.join(MODEL_DIR, "v8_winner_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "v8_winner_scaler.pkl"))

print("\n💾 Model saved.")
print("🏆 v8 WINNER EDITION READY.")