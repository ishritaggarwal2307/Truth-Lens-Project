print(">>> SCRIPT STARTED")

import numpy as np
from pathlib import Path

# ===============================
# PATH CONFIGURATION
# ===============================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

print(">>> BASE_DIR:", BASE_DIR)
print(">>> RAW_DIR:", RAW_DIR)
print(">>> PROCESSED_DIR:", PROCESSED_DIR)

# ===============================
# CREATE REQUIRED FOLDERS
# ===============================
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# CREATE DUMMY DATA (DAY-0 SAFE)
# ===============================
print(">>> Creating dummy training data")

# 200 grayscale images of size 64x64
X = np.random.rand(200, 64, 64, 1).astype("float32")
y = np.random.randint(0, 2, size=(200,))

# ===============================
# SAVE DATA
# ===============================
np.save(PROCESSED_DIR / "X.npy", X)
np.save(PROCESSED_DIR / "y.npy", y)

print(">>> Dummy X.npy and y.npy created")
print(">>> X shape:", X.shape)
print(">>> y shape:", y.shape)
print(">>> SCRIPT FINISHED SUCCESSFULLY")