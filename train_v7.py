import os
import random
import numpy as np
import librosa
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier

# =========================
# GLOBAL CONFIG
# =========================
SEED = 42
SAMPLE_RATE = 22050

random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "audio")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "v7_1_elite_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "v7_1_elite_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "v7_1_feature_meta.pkl")

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

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    for file in os.listdir(folder):
        if not file.lower().endswith((".wav", ".mp3", ".flac")):
            continue

        file_path = os.path.join(folder, file)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"⚠ Skipping {file}: {e}")

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples")

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL (Tuned)
# =========================
print("🚀 Training Elite XGBoost Model...")

model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_lambda=2,
    eval_metric="logloss",
    random_state=SEED,
    use_label_encoder=False
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n📊 ELITE PERFORMANCE REPORT")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC:", round(roc_auc, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# =========================
# SHAP EXPLAINABILITY
# =========================
print("\n🧠 Generating SHAP Explainability...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Global Feature Importance")
plt.savefig(os.path.join(MODEL_DIR, "shap_global.png"))
plt.close()

# Local explanation (first test sample)
plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test[0],
    matplotlib=True,
    show=False
)
plt.title("SHAP Local Explanation (Sample 0)")
plt.savefig(os.path.join(MODEL_DIR, "shap_local_sample0.png"))
plt.close()

print("✅ SHAP visualizations saved.")

# =========================
# SAVE EVERYTHING
# =========================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

feature_meta = {
    "categories": CATEGORIES,
    "feature_length": X.shape[1]
}

joblib.dump(feature_meta, META_PATH)

print("\n💾 Model, scaler, and metadata saved.")
print("🏆 v7.1 ELITE training complete.")