print(">>> TRAINING SCRIPT STARTED")

# ===============================
# IMPORTS
# ===============================
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print(">>> Imports successful")

# ===============================
# PATH CONFIGURATION
# ===============================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

print(">>> PROCESSED_DIR:", PROCESSED_DIR)
print(">>> MODEL_DIR:", MODEL_DIR)

# ===============================
# LOAD DATA
# ===============================
print(">>> Loading processed data")

X = np.load(PROCESSED_DIR / "X.npy")
y = np.load(PROCESSED_DIR / "y.npy")

print(">>> Data loaded")
print(">>> X shape:", X.shape)
print(">>> y shape:", y.shape)

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# BUILD CNN MODEL
# ===============================
print(">>> Building CNN model")

model = keras.Sequential([
    layers.Input(shape=X_train.shape[1:]),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(">>> Model compiled")

# ===============================
# TRAIN MODEL
# ===============================
print(">>> Starting training")

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=16
)

# ===============================
# SAVE MODEL
# ===============================
MODEL_DIR.mkdir(exist_ok=True)
model.save(str(MODEL_DIR / "cnn_model.h5"))

print(">>> TRAINING COMPLETE")