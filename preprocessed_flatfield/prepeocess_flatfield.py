import os
import cv2
import csv
import numpy as np
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

DATASET_FLATFIELD = "Data/Flatfield"
OUTPUT_DIR = "processed_data/Flatfield"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "C:/Users/ASUS/Downloads/ai/Data/Official/metadata_features.csv")

def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0  # normalize [0,1]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def extract_noise_residual(img):
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    return img - denoised

def compute_metadata_features(img, file_name, scanner_id):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_name) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "file_name": os.path.basename(file_name),
        "main_class": "Flatfield",
        "resolution": "N/A",   # Flatfield has no 150/300 subfolders
        "class_label": scanner_id,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def compute_flatfield_fingerprints(flatfield_dir, out_dir, csv_path):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "file_name", "main_class", "resolution", "class_label",
            "width", "height", "aspect_ratio", "file_size_kb",
            "mean_intensity", "std_intensity", "skewness", "kurtosis",
            "entropy", "edge_density"
        ])
        writer.writeheader()

        for scanner_id in os.listdir(flatfield_dir):
            scanner_path = os.path.join(flatfield_dir, scanner_id)
            if not os.path.isdir(scanner_path):
                continue

            residuals = []
            for file in os.listdir(scanner_path):
                img_path = os.path.join(scanner_path, file)
                if not file.lower().endswith(('.png', '.tif', '.jpg', '.jpeg')):
                    print(f"⚠️ Skipping non-image file: {file}")
                    continue

                try:
                    img = load_and_preprocess(img_path)
                    residual = extract_noise_residual(img)
                    residuals.append(residual)

                    # Metadata
                    features = compute_metadata_features(img, img_path, scanner_id)
                    writer.writerow(features)

                except Exception as e:
                    print(f"⚠️ Skipping {img_path} due to error: {e}")
                    continue

            if residuals:
                fingerprint = np.mean(residuals, axis=0)
                np.save(os.path.join(out_dir, f"{scanner_id}_fingerprint.npy"), fingerprint)
                print(f"✅ Saved fingerprint for {scanner_id}")

if __name__ == "__main__":
    compute_flatfield_fingerprints(DATASET_FLATFIELD, OUTPUT_DIR, CSV_PATH)
    print("🎯 Flatfield preprocessing + metadata feature extraction complete.")