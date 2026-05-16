import os
import random
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
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
# K-FOLD CROSS VALIDATION
# =========================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

accuracies = []
roc_aucs = []

fold = 1

for train_idx, test_idx in skf.split(X, y):

    print(f"\n📊 Fold {fold}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        reg_lambda=2,
        eval_metric="logloss",
        random_state=SEED,
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    accuracies.append(acc)
    roc_aucs.append(roc_auc)

    print("Accuracy:", round(acc * 100, 2), "%")
    print("ROC-AUC:", round(roc_auc, 4))

    fold += 1

# =========================
# FINAL METRICS
# =========================
print("\n🏆 FINAL CROSS-VALIDATION RESULTS")
print("Mean Accuracy:", round(np.mean(accuracies) * 100, 2), "%")
print("Std Accuracy:", round(np.std(accuracies) * 100, 2), "%")
print("Mean ROC-AUC:", round(np.mean(roc_aucs), 4))

# =========================
# TRAIN FINAL MODEL ON FULL DATA
# =========================
print("\n🚀 Training final model on full dataset...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_lambda=2,
    eval_metric="logloss",
    random_state=SEED,
    use_label_encoder=False
)

final_model.fit(X_scaled, y)

# =========================
# SAVE MODEL
# =========================
joblib.dump(final_model, os.path.join(MODEL_DIR, "v7_3_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "v7_3_scaler.pkl"))

print("\n💾 Model saved.")
print("🎯 v7.3 Robust Evaluation Complete.")