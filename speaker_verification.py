import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SAMPLE_RATE = 22050
N_MFCC = 60


def extract_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    embedding = np.concatenate([mfcc_mean, mfcc_std])
    return embedding


def verify_speaker(reference_path, test_path, threshold=0.80):
    emb1 = extract_embedding(reference_path)
    emb2 = extract_embedding(test_path)

    similarity = cosine_similarity(
        emb1.reshape(1, -1),
        emb2.reshape(1, -1)
    )[0][0]

    return similarity >= threshold, float(similarity)