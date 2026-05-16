import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

DATASET_PATH = "data/audio"
MODEL_SAVE_PATH = "models/truth_lens_v6.pkl"
SAMPLE_RATE = 22050


# ===============================
# Advanced Feature Extraction
# ===============================

def extract_features(audio, sr):

    # MFCC (mean + std)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    # Log-Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel)
    log_mel_mean = np.mean(log_mel.T, axis=0)

    # Spectral flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

    # Harmonic-Percussive Ratio
    harmonic, percussive = librosa.effects.hpss(audio)
    harm_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6)

    # ZCR & RMS
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([
        mfcc_mean, mfcc_std,
        chroma_mean,
        contrast_mean,
        log_mel_mean,
        flatness,
        harm_ratio,
        zcr,
        rms
    ])


# ===============================
# Data Augmentation
# ===============================

def augment_audio(audio, sr):

    augmented = []

    # Add noise
    noise = audio + 0.005 * np.random.randn(len(audio))
    augmented.append(noise)

    # Pitch shift
    pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    augmented.append(pitch)

    # Time stretch
    stretch = librosa.effects.time_stretch(audio, rate=1.1)
    augmented.append(stretch)

    return augmented


# ===============================
# Load Dataset
# ===============================

def load_dataset():
    features = []
    labels = []

    for label in ["real", "fake"]:
        folder = os.path.join(DATASET_PATH, label)
        print(f"\nProcessing {label} samples...")

        for file in tqdm(os.listdir(folder)):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                audio, sr = librosa.load(path, sr=SAMPLE_RATE)

                label_id = 1 if label == "fake" else 0

                # Original
                features.append(extract_features(audio, sr))
                labels.append(label_id)

                # Augmented
                for aug in augment_audio(audio, sr):
                    features.append(extract_features(aug, sr))
                    labels.append(label_id)

    return np.array(features), np.array(labels)


# ===============================
# Train Models
# ===============================

def main():

    print("Loading dataset...")
    X, y = load_dataset()

    print("Total samples:", len(X))
    print("Feature dimension:", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300)
    rf.fit(X_train, y_train)

    print("Training LightGBM...")
    lgb = LGBMClassifier(n_estimators=400)
    lgb.fit(X_train, y_train)

    # Weighted Ensemble
    xgb_prob = xgb.predict_proba(X_test)
    rf_prob = rf.predict_proba(X_test)
    lgb_prob = lgb.predict_proba(X_test)

    ensemble_prob = (
        0.5 * xgb_prob +
        0.3 * rf_prob +
        0.2 * lgb_prob
    )

    y_pred = np.argmax(ensemble_prob, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy * 100)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump((xgb, rf, lgb, scaler), MODEL_SAVE_PATH)

    print("\n✅ v6 Model Saved Successfully.")


if __name__ == "__main__":
    main()