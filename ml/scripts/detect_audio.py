from pathlib import Path
import numpy as np
import librosa

# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
AUDIO_DIR = BASE_DIR / "data" / "audio"
REAL_DIR = AUDIO_DIR / "real"
FAKE_DIR = AUDIO_DIR / "fake"

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=5)

        if len(y) == 0:
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        return np.array([
            np.mean(np.abs(mfcc)),
            np.var(mfcc),
            np.mean(zcr),
            np.mean(centroid)
        ])

    except:
        return None


# =====================================================
# BASELINE LEARNING FROM REAL SAMPLES
# =====================================================

def learn_human_baseline():
    real_features = []

    for audio_file in REAL_DIR.glob("*"):
        if audio_file.suffix.lower() not in [".wav", ".mp3", ".m4a"]:
            continue

        features = extract_features(audio_file)
        if features is not None:
            real_features.append(features)

    if len(real_features) == 0:
        return None, None

    real_features = np.array(real_features)

    mean_vector = np.mean(real_features, axis=0)
    std_vector = np.std(real_features, axis=0) + 1e-6  # avoid divide by zero

    return mean_vector, std_vector


# =====================================================
# STATISTICAL DISTANCE SCORING
# =====================================================

def predict_fake_probability(features, mean_vector, std_vector):

    if features is None or mean_vector is None:
        return 0.5

    # Z-score distance
    z_scores = np.abs((features - mean_vector) / std_vector)

    distance = np.mean(z_scores)

    # Soft logistic scaling
    scaled = (distance - 1.2) * 2
    fake_probability = 1 / (1 + np.exp(-scaled))

    return float(np.clip(fake_probability, 0, 1))


# =====================================================
# STREAMLIT FUNCTION
# =====================================================

def analyze_and_return(folder_path, label):

    results = []

    mean_vector, std_vector = learn_human_baseline()

    for audio_file in folder_path.glob("*"):

        if audio_file.suffix.lower() not in [".wav", ".mp3", ".m4a"]:
            continue

        features = extract_features(audio_file)
        fake_score = predict_fake_probability(features, mean_vector, std_vector)

        results.append({
            "file": audio_file.name,
            "fake_score": round(fake_score, 2),
            "real_score": round(1 - fake_score, 2)
        })

    return results


# =====================================================
# CLI MODE
# =====================================================

def analyze_folder(folder, label):

    mean_vector, std_vector = learn_human_baseline()

    print(f"\nAnalyzing {label} samples:")

    for audio_file in folder.glob("*"):

        if audio_file.suffix.lower() not in [".wav", ".mp3", ".m4a"]:
            continue

        features = extract_features(audio_file)
        fake_score = predict_fake_probability(features, mean_vector, std_vector)

        print(f"\nFile: {audio_file.name}")
        print(f"→ Fake Probability: {fake_score:.2f}")
        print(f"→ Real Probability: {1 - fake_score:.2f}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print(">>> TRUTH LENS – ADAPTIVE BASELINE DETECTOR STARTED")

    analyze_folder(REAL_DIR, "REAL")
    analyze_folder(FAKE_DIR, "FAKE")

    print("\n>>> Detection complete")