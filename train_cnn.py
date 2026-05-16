import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf

DATA_PATH = "data/audio"
SAMPLE_RATE = 22050
IMG_SIZE = 128


def create_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = tf.image.resize(spec_db[..., np.newaxis], [IMG_SIZE, IMG_SIZE])
    return spec_db.numpy()


X = []
y = []

print("Generating spectrogram dataset...")

for label, folder in enumerate(["real", "fake"]):
    folder_path = os.path.join(DATA_PATH, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        spec = create_spectrogram(file_path)
        X.append(spec)
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print("CNN Accuracy:", acc)

model.save("models/truth_lens_cnn.h5")