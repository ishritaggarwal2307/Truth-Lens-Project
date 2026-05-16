import os
import librosa
import numpy as np
import joblib
from train_model import extract_features

MODEL_PATH = "models/truth_lens_elite.pkl"
SAMPLE_RATE = 22050
CROSS_PATH = "data/cross_test"

xgb_model, rf_model, scaler = joblib.load(MODEL_PATH)

def evaluate_folder(folder, label):
    correct = 0
    total = 0

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)

            features = extract_features(audio, sr)
            features = scaler.transform([features])

            xgb_prob = xgb_model.predict_proba(features)[0]
            rf_prob = rf_model.predict_proba(features)[0]
            prob = (xgb_prob + rf_prob) / 2

            prediction = np.argmax(prob)

            if prediction == label:
                correct += 1
            total += 1

    return correct, total


real_correct, real_total = evaluate_folder(
    os.path.join(CROSS_PATH, "real"), 0
)

fake_correct, fake_total = evaluate_folder(
    os.path.join(CROSS_PATH, "fake"), 1
)

accuracy = (real_correct + fake_correct) / (real_total + fake_total)

print("\nCross-Dataset Accuracy:", accuracy * 100)
print("Real Accuracy:", real_correct / real_total * 100)
print("Fake Accuracy:", fake_correct / fake_total * 100)