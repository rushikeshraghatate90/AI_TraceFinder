# train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from build_cnn import build_dual_branch
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

# ----------------------
# Load data
# ----------------------
X_img = np.load("processed_data/new_processed/official_images.npy")
X_noise = np.load("processed_data/new_processed/flatfield_noise.npy")
y = np.load("processed_data/new_processed/labels.npy")

print("Before processing:")
print("X_img shape:", X_img.shape)
print("X_noise shape:", X_noise.shape)
print("y shape:", y.shape)

# ----------------------
# Convert noise to grayscale if needed
# ----------------------
if X_noise.shape[-1] == 3:
    X_noise = np.mean(X_noise, axis=-1, keepdims=True)

print("After grayscale conversion of X_noise:")
print("X_noise shape:", X_noise.shape)

# ----------------------
# Upsample noise to match image length
# ----------------------
if len(X_noise) < len(X_img):
    reps = len(X_img) // len(X_noise) + 1
    X_noise = np.tile(X_noise, (reps, 1, 1, 1))[:len(X_img)]

print("After upsampling X_noise:")
print("X_img shape:", X_img.shape)
print("X_noise shape:", X_noise.shape)
print("y shape:", y.shape)

# ----------------------
# Normalize inputs
# ----------------------
X_img = X_img.astype("float32") / 255.0
X_noise = X_noise.astype("float32") / 255.0

# ----------------------
# Train/validation split
# ----------------------
X_img_train, X_img_val, X_noise_train, X_noise_val, y_train, y_val = train_test_split(
    X_img, X_noise, y, test_size=0.2, random_state=42
)

# ----------------------
# Build dual-branch CNN
# ----------------------
model = build_dual_branch(num_classes=len(np.unique(y)))

# ----------------------
# Callbacks
# ----------------------
os.makedirs("models", exist_ok=True)

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint("models/best_dual_branch.keras", save_best_only=True),
    CSVLogger("models/training_log.csv")
]

# ----------------------
# Train model
# ----------------------
history = model.fit(
    {"official_input": X_img_train, "noise_input": X_noise_train},
    y_train,
    validation_data=(
        {"official_input": X_img_val, "noise_input": X_noise_val}, y_val
    ),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# ----------------------
# Save final model
# ----------------------
model.save("models/final_dual_branch.keras")
print("✅ Model training complete! Best and final models saved in /models/")
    