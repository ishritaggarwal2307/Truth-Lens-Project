print(">>> FEATURE EXTRACTION STARTED")

import os
import numpy as np
import librosa
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
AUDIO_DIR = BASE_DIR / "data" / "audio"
FEATURE_DIR = BASE_DIR / "data" / "processed"

FEATURE_DIR.mkdir(exist_ok=True)

X = []
y = []

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)

print(">>> Processing REAL audio")
for file in (AUDIO_DIR / "real").glob("*.wav"):
    X.append(extract_mfcc(file))
    y.append(0)

print(">>> Processing FAKE audio")
for file in (AUDIO_DIR / "fake").glob("*.wav"):
    X.append(extract_mfcc(file))
    y.append(1)

X = np.array(X)
y = np.array(y)

np.save(FEATURE_DIR / "audio_X.npy", X)
np.save(FEATURE_DIR / "audio_y.npy", y)

print(">>> Feature extraction completed")
print(">>> X shape:", X.shape)
print(">>> y shape:", y.shape)