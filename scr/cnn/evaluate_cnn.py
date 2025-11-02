import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Load model + data
model = tf.keras.models.load_model("processed/dual_branch_cnn.h5")
X_img = np.load("processed_data/new_preprocessed/official_images.npy")
X_noise = np.load("processed_data/new_preprocessed/flatfield_noise.npy")
y = np.load("processed_data/new_preprocessed/labels.npy")

# Predict
y_pred = model.predict({"official_input": X_img, "noise_input": X_noise})
y_pred_classes = np.argmax(y_pred, axis=1)

print("Classification Report:")
print(classification_report(y, y_pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred_classes))