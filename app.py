import streamlit as st
import cv2
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import joblib
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random
import hashlib
from tabulate import tabulate
import pickle
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy import ndimage
from scipy.fft import fft2, fftshift
import pywt
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import csv
import glob
import subprocess
import re

# --- Project Root Directory ---
# Defines the absolute path to the project root, making the app portable.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def set_app_style():
    '''Applies custom CSS styles to the Streamlit application.'''
    st.markdown(r'''
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
    ''', unsafe_allow_html=True)

@st.cache_resource
def load_hybrid_model_artifacts():
    ART_DIR = os.path.join(ROOT_DIR, "proceed_data")
    CKPT_PATH = os.path.join(ART_DIR, "scanner_hybrid_final.keras")
    hyb_model = tf.keras.models.load_model(CKPT_PATH, compile=False)

    with open(os.path.join(ART_DIR, "hybrid_label_encoder.pkl"), "rb") as f:
        le_inf = pickle.load(f)

    with open(os.path.join(ART_DIR, "hybrid_feat_scaler.pkl"), "rb") as f:
        scaler_inf = pickle.load(f)
    
    FP_PATH = os.path.join(ART_DIR, "Flatfield/scanner_fingerprints.pkl")
    with open(FP_PATH, "rb") as f:
        scanner_fps_inf = pickle.load(f)

    ORDER_NPY = os.path.join(ART_DIR, "Flatfield/fp_keys.npy")
    fp_keys_inf = np.load(ORDER_NPY, allow_pickle=True).tolist()
    
    return hyb_model, le_inf, scaler_inf, scanner_fps_inf, fp_keys_inf

def predict_with_hybrid_model(image_path):
    hyb_model, le_inf, scaler_inf, scanner_fps_inf, fp_keys_inf = load_hybrid_model_artifacts()
    
    IMG_SIZE = (256, 256)

    def corr2d(a, b):
        a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / d) if d != 0 else 0.0

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape; cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6
        bins = np.linspace(0, rmax, K+1)
        return [float(mag[(r >= bins[i]) & (r < bins[i+1])].mean() if ((r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
        g8 = (g * 255.0).astype(np.uint8)
        codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
        return hist.astype(np.float32).tolist()

    def preprocess_residual_pywt(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        cH.fill(0); cV.fill(0); cD.fill(0)
        den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        return (img - den).astype(np.float32)

    def make_feats_from_res(res):
        v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
        v_fft  = fft_radial_energy(res)
        v_lbp  = lbp_hist_safe(res)
        v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
        return scaler_inf.transform(v)

    res = preprocess_residual_pywt(image_path)
    x_img = np.expand_dims(res, axis=(0,-1))
    x_feat = make_feats_from_res(res)
    
    prob = hyb_model.predict([x_img, x_feat], verbose=0)[0]
    idx = int(np.argmax(prob))
    label = le_inf.classes_[idx]
    conf = float(prob[idx] * 100)
    
    return label, conf, prob, le_inf.classes_

def predict_with_baseline_model(image_path, model_choice="rf"):
    # Load artifacts
    _, le_inf, _, scanner_fps_inf, fp_keys_inf = load_hybrid_model_artifacts()
    scaler = joblib.load(os.path.join(ROOT_DIR, "models", "scaler.pkl"))
    
    # Load the selected baseline model
    if model_choice == "rf":
        model = joblib.load(os.path.join(ROOT_DIR, "models", "random_forest.pkl"))
    else:
        model = joblib.load(os.path.join(ROOT_DIR, "models", "svm.pkl"))

    IMG_SIZE = (256, 256)

    # Feature extraction functions (nested for encapsulation)
    def corr2d(a, b):
        a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / d) if d != 0 else 0.0

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape; cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6
        bins = np.linspace(0, rmax, K+1)
        return [float(mag[(r >= bins[i]) & (r < bins[i+1])].mean() if ((r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
        g8 = (g * 255.0).astype(np.uint8)
        codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
        return hist.astype(np.float32).tolist()

    def preprocess_residual_pywt(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        cH.fill(0); cV.fill(0); cD.fill(0)
        den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        return (img - den).astype(np.float32)

    def make_feats_from_res(res):
        v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
        v_fft  = fft_radial_energy(res)
        v_lbp  = lbp_hist_safe(res)
        v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
        return scaler.transform(v)

    # Main prediction logic
    res = preprocess_residual_pywt(image_path)
    features = make_feats_from_res(res)
    
    # Predict
    pred_idx = model.predict(features)[0]
    pred_label = le_inf.classes_[pred_idx]  # Convert numerical label to scanner name
    
    # Get probabilities
    prob = model.predict_proba(features)[0]
    conf = float(np.max(prob) * 100)
    
    return pred_label, conf, prob, le_inf.classes_

def run_eda(dataset_path):
    records = []
    corrupted_files = []
    duplicate_files = []
    hashes = {}

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for root, dirs, files in os.walk(class_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                    file_path = os.path.join(root, f)
                    img = cv2.imread(file_path)
                    if img is None:
                        corrupted_files.append(file_path)
                        continue

                    # Check for duplicates
                    try:
                        with open(file_path, "rb") as f_img:
                            img_hash = hashlib.md5(f_img.read()).hexdigest()
                        if img_hash in hashes:
                            duplicate_files.append(file_path)
                        else:
                            hashes[img_hash] = file_path
                    except Exception as e:
                        st.warning(f"Could not hash {file_path}: {e}")

                    h, w = img.shape[:2]
                    brightness = img.mean()
                    
                    # Extract resolution from path using a more flexible regex
                    resolution = "Unknown"
                    match = re.search(r'(150|300|600)', root)
                    if match:
                        resolution = f"{match.group(1)} DPI"
                    
                    records.append({
                        "file": file_path,
                        "scanner": class_name,
                        "height": h,
                        "width": w,
                        "brightness": brightness,
                        "resolution": resolution
                    })

    df = pd.DataFrame(records)
    return df, corrupted_files, duplicate_files

def preprocess_and_compute_residuals():
    OFFICIAL_DIR = os.path.join(ROOT_DIR, "proceed_data", "official")
    WIKI_DIR = os.path.join(ROOT_DIR, "proceed_data", "Wikipedia")
    OUT_PATH = os.path.join(ROOT_DIR, "proceed_data", "official_wiki_residuals.pkl")

    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def resize_to(img, size=(256, 256)):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def normalize_img(img):
        return img.astype(np.float32) / 255.0

    def denoise_wavelet(img):
        coeffs = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs
        cH[:] = 0
        cV[:] = 0
        cD[:] = 0
        return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

    def compute_residual(img):
        denoised = denoise_wavelet(img)
        return img - denoised

    def process_single_image(fpath):
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray = to_gray(img)
        gray = resize_to(gray, (256, 256))
        gray = normalize_img(gray)
        return compute_residual(gray)

    def process_dataset(base_dir, dataset_name, residuals_dict):
        st.write(f"Recursively preprocessing {dataset_name} images...")
        dpi_dirs_to_process = []
        for root, dirs, files in os.walk(base_dir):
            if os.path.basename(root) in ['150', '300']:
                dpi_dirs_to_process.append(root)

        if not dpi_dirs_to_process:
            st.warning(f"Warning: No '150' or '300' DPI subfolders found in '{base_dir}'.")
            return

        for dpi_path in tqdm(dpi_dirs_to_process, desc=f"Processing {dataset_name} DPI folders"):
            dpi = os.path.basename(dpi_path)
            scanner_name = os.path.basename(os.path.dirname(dpi_path))

            files = [
                os.path.join(dpi_path, f) 
                for f in os.listdir(dpi_path) 
                if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))
            ]
            
            if not files:
                continue

            dpi_residuals = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_single_image, f) for f in files]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        dpi_residuals.append(res)
            
            if scanner_name not in residuals_dict[dataset_name]:
                residuals_dict[dataset_name][scanner_name] = {}
            if dpi not in residuals_dict[dataset_name][scanner_name]:
                residuals_dict[dataset_name][scanner_name][dpi] = []
            
            residuals_dict[dataset_name][scanner_name][dpi].extend(dpi_residuals)

    residuals_dict = {"Official": {}, "Wikipedia": {}}
    process_dataset(OFFICIAL_DIR, "Official", residuals_dict)
    process_dataset(WIKI_DIR, "Wikipedia", residuals_dict)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(residuals_dict, f)

    st.write(f"Saved Official + Wikipedia residuals (150 & 300 DPI separately) to {OUT_PATH}")

def compute_scanner_fingerprints():
    FLATFIELD_RESIDUALS_PATH = os.path.join(ROOT_DIR, "proceed_data", "flatfield_residuals.pkl")
    FP_OUT_PATH = os.path.join(ROOT_DIR, "proceed_data", "Flatfield", "scanner_fingerprints.pkl")
    ORDER_NPY = os.path.join(ROOT_DIR, "proceed_data", "Flatfield", "fp_keys.npy")

    with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
        flatfield_residuals = pickle.load(f)

    scanner_fingerprints = {}
    st.write("🔄 Computing fingerprints from Flatfields...")
    for scanner, residuals in flatfield_residuals.items():
        if not residuals:
            continue
        stack = np.stack(residuals, axis=0)
        fingerprint = np.mean(stack, axis=0)
        scanner_fingerprints[scanner] = fingerprint

    with open(FP_OUT_PATH, "wb") as f:
        pickle.dump(scanner_fingerprints, f)

    fp_keys = sorted(scanner_fingerprints.keys())
    np.save(ORDER_NPY, np.array(fp_keys))
    st.write(f"✅ Saved {len(scanner_fingerprints)} fingerprints and fp_keys.npy")

def extract_prnu_features():
    FP_OUT_PATH = os.path.join(ROOT_DIR, "proceed_data", "Flatfield", "scanner_fingerprints.pkl")
    ORDER_NPY = os.path.join(ROOT_DIR, "proceed_data", "Flatfield", "fp_keys.npy")
    RES_PATH = os.path.join(ROOT_DIR, "proceed_data", "official_wiki_residuals.pkl")
    FEATURES_OUT = os.path.join(ROOT_DIR, "proceed_data", "features.pkl")

    with open(FP_OUT_PATH, "rb") as f:
        scanner_fingerprints = pickle.load(f)
    fp_keys = np.load(ORDER_NPY)

    def corr2d(a, b):
        a = a.astype(np.float32).ravel()
        b = b.astype(np.float32).ravel()
        a -= a.mean()
        b -= b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float((a @ b) / denom) if denom != 0 else 0.0

    with open(RES_PATH, "rb") as f:
        residuals_dict = pickle.load(f)

    features, labels = [], []
    for dataset_name in ["Official", "Wikipedia"]:
        st.write(f"🔄 Computing PRNU features for {dataset_name} ...")
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    vec = [corr2d(res, scanner_fingerprints[k]) for k in fp_keys]
                    features.append(vec)
                    labels.append(scanner)

    with open(FEATURES_OUT, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)
    st.write(f"✅ Saved features shape: {len(features)} x {len(features[0])}")

def extract_enhanced_features_func(): # Renamed to avoid conflict
    RES_PATH = os.path.join(ROOT_DIR, "proceed_data", "official_wiki_residuals.pkl")
    ENHANCED_OUT = os.path.join(ROOT_DIR, "proceed_data", "enhanced_features.pkl")

    def extract_enhanced_features(residual):
        fft_img = np.abs(fft2(residual))
        fft_img = fftshift(fft_img)
        h, w = fft_img.shape
        center_h, center_w = h//2, w//2
        low_freq = np.mean(fft_img[center_h-20:center_h+20, center_w-20:center_w+20])
        mid_freq = np.mean(fft_img[center_h-60:center_h+60, center_w-60:center_w+60]) - low_freq
        high_freq = np.mean(fft_img) - low_freq - mid_freq

        res_range = np.max(residual) - np.min(residual)
        if res_range > 0:
            residual_uint8 = (255 * (residual - np.min(residual)) / res_range).astype(np.uint8)
        else:
            residual_uint8 = np.zeros_like(residual, dtype=np.uint8)
        lbp = local_binary_pattern(residual_uint8, P=24, R=3, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 25), density=True)

        grad_x = ndimage.sobel(residual, axis=1)
        grad_y = ndimage.sobel(residual, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        texture_features = [
            np.std(residual),
            np.mean(np.abs(residual)),
            np.std(grad_mag),
            np.mean(grad_mag)
        ]

        return [low_freq, mid_freq, high_freq] + lbp_hist.tolist() + texture_features

    with open(RES_PATH, "rb") as f:
        residuals_dict = pickle.load(f)

    enhanced_features, enhanced_labels = [], []
    for dataset_name in ["Official", "Wikipedia"]:
        st.write(f"🔄 Extracting enhanced features for {dataset_name} ...")
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    feat = extract_enhanced_features(res)
                    enhanced_features.append(feat)
                    enhanced_labels.append(scanner)

    with open(ENHANCED_OUT, "wb") as f:
        pickle.dump({"features": enhanced_features, "labels": enhanced_labels}, f)
    st.write(f"✅ Enhanced features shape: {len(enhanced_features)} x {len(enhanced_features[0])}")
    st.write(f"✅ Saved enhanced features to {ENHANCED_OUT}")

def predict_folder(folder_path, output_csv="hybrid_folder_results.csv"):
    IMG_SIZE = (256, 256)
    ART_DIR = os.path.join(ROOT_DIR, "proceed_data")
    FP_PATH = os.path.join(ART_DIR, "Flatfield/scanner_fingerprints.pkl")
    ORDER_NPY = os.path.join(ART_DIR, "Flatfield/fp_keys.npy")
    CKPT_PATH = os.path.join(ART_DIR, "scanner_hybrid_final.keras")
    output_csv_path = os.path.join(ROOT_DIR, output_csv)

    hyb_model = tf.keras.models.load_model(CKPT_PATH, compile=False)

    with open(os.path.join(ART_DIR, "hybrid_label_encoder.pkl"), "rb") as f:
        le_inf = pickle.load(f)

    with open(os.path.join(ART_DIR, "hybrid_feat_scaler.pkl"), "rb") as f:
        scaler_inf = pickle.load(f)

    with open(FP_PATH, "rb") as f:
        scanner_fps_inf = pickle.load(f)

    fp_keys_inf = np.load(ORDER_NPY, allow_pickle=True).tolist()

    def corr2d(a, b):
        a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / d) if d != 0 else 0.0

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape; cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6
        bins = np.linspace(0, rmax, K+1)
        return [float(mag[(r >= bins[i]) & (r < bins[i+1])].mean() if ((r >= bins[i]) & (r < bins[i+1])).any() else 0.0) for i in range(K)]

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
        g8 = (g * 255.0).astype(np.uint8)
        codes = local_binary_pattern(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
        return hist.astype(np.float32).tolist()

    def preprocess_residual_pywt(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        cH.fill(0); cV.fill(0); cD.fill(0)
        den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        return (img - den).astype(np.float32)

    def make_feats_from_res(res):
        v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
        v_fft  = fft_radial_energy(res)
        v_lbp  = lbp_hist_safe(res)
        v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
        return scaler_inf.transform(v)

    def predict_scanner_hybrid(image_path):
        res = preprocess_residual_pywt(image_path)
        x_img = np.expand_dims(res, axis=(0,-1))
        x_feat = make_feats_from_res(res)
        prob = hyb_model.predict([x_img, x_feat], verbose=0)
        idx = int(np.argmax(prob))
        label = le_inf.classes_[idx]
        conf = float(prob[0, idx]*100)
        return label, conf

    exts=("*.tif","*.png","*.jpg","*.jpeg")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    st.write(f"Found {len(image_files)} images in {folder_path}")

    results = []
    for img_path in image_files:
        try:
            label, conf = predict_scanner_hybrid(img_path)
            results.append((img_path, label, conf))
            st.write(f"{img_path} -> {label} | {conf:.2f}%")
        except Exception as e:
            st.error(f"⚠️ Error {img_path}: {e}")

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Predicted_Label", "Confidence (%)"])
        writer.writerows(results)
    st.write(f"\n✅ Predictions saved to {output_csv_path}")
    return results

def home():
    """Renders the Home page of the application."""
    st.title("TraceFinder – Forensic Scanner Identification")

    st.markdown("""
    **Purpose:** Identify the source scanner device used to scan a document/image by analyzing scanner-specific artifacts.

    Use this app to explore dataset statistics, visualize features (noise / FFT / PRNU-like maps), view model training results, and run a live prediction on an uploaded scan.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        # --- Quick Stats ---
        st.header("Quick Stats ⚙️")
        
        # Calculate stats
        total_images = 0
        proceed_data_path = os.path.join(ROOT_DIR, "proceed_data")
        if os.path.exists(proceed_data_path):
            for root, dirs, files in os.walk(proceed_data_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        total_images += 1

        le_path = os.path.join(ROOT_DIR, "proceed_data", "hybrid_label_encoder.pkl")
        scanner_classes = 0
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                le = pickle.load(f)
                scanner_classes = len(le.classes_)

        st.metric("Total Images", f"{total_images}")
        st.metric("Scanner Classes", f"{scanner_classes}")
        st.metric("DPI Levels", "150, 300, 600")

    with col2:
        # --- Banner Image ---
        banner_image_path = os.path.join(ROOT_DIR, "images", "AI_TraceFinder.png")
        if os.path.exists(banner_image_path):
            st.image(banner_image_path, width='stretch')
        else:
            st.warning("Banner image not found. Please add a 'AI_TraceFinder.png' to the 'images' directory.")

def prediction():
    """Renders the Prediction page."""
    st.title("Scanner Identification")
    st.write("Upload an image to identify the scanner model.")

    model_choice = st.selectbox("Choose a model", ["Hybrid CNN", "Random Forest", "SVM"])

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tif"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', width='stretch')
        if st.button("Predict Scanner Model"):
            
            model_choice_short = ""
            if model_choice == "Random Forest":
                model_choice_short = "rf"
            elif model_choice == "SVM":
                model_choice_short = "svm"

            with st.spinner(f'Analyzing image and predicting with {model_choice}...'):
                try:
                    temp_path = os.path.join(ROOT_DIR, "temp_image.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if model_choice == "Hybrid CNN":
                        pred, conf, prob, classes = predict_with_hybrid_model(temp_path)
                    else:
                        pred, conf, prob, classes = predict_with_baseline_model(temp_path, model_choice=model_choice_short)

                    st.success(f"Predicted Scanner Model: **{pred}**")
                    st.metric(label="Confidence", value=f"{conf:.2f}%")
                    st.info(f"Prediction made using: **{model_choice}**")

                    st.write("**Top Prediction Probability:**")
                    prob_df = pd.DataFrame({
                        'Scanner Model': classes,
                        'Probability': prob
                    })
                    st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}))

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

def folder_analysis():
    st.title("Folder Analysis Dashboard")
    st.write("This dashboard provides an overview of the project's artifacts.")

    # --- Datasets ---
    st.header("Datasets")
    
    proceed_data_path = os.path.join(ROOT_DIR, "proceed_data")
    processed_data_path = os.path.join(ROOT_DIR, "processed_data")

    tab1, tab2 = st.tabs(["Raw Data", "Processed Data"])

    with tab1:
        st.subheader("Raw Data")
        if os.path.exists(proceed_data_path):
            datasets = [d for d in os.listdir(proceed_data_path) if os.path.isdir(os.path.join(proceed_data_path, d))]
            selected_dataset = st.selectbox("Select a dataset to inspect", datasets)
            if selected_dataset:
                dataset_path = os.path.join(proceed_data_path, selected_dataset)
                files = []
                for root, _, fls in os.walk(dataset_path):
                    for f in fls:
                        files.append(os.path.join(root, f))
                
                st.write(f"Found {len(files)} files in {selected_dataset}.")
                
                if files:
                    file_to_view = st.selectbox("Select a file to view", files, key="raw_data_viewer")
                    if file_to_view:
                        file_type = os.path.splitext(file_to_view)[1].lower()
                        if file_type in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                            if os.path.exists(file_to_view):
                                st.image(file_to_view, caption=f"Viewing: {file_to_view}")
                            else:
                                st.warning(f"Image file not found: {file_to_view}")
                        else:
                            st.info(f"File type {file_type} cannot be previewed here.")
        else:
            st.warning("`proceed_data` directory not found.")

    with tab2:
        st.subheader("Processed Data")
        if os.path.exists(processed_data_path):
            processed_files = []
            for root, _, fls in os.walk(processed_data_path):
                for f in fls:
                    processed_files.append(os.path.join(root, f))
            
            st.write(f"Found {len(processed_files)} processed files.")
            if processed_files:
                file_to_view = st.selectbox("Select a processed file to inspect", processed_files, key="processed_data_viewer")
                if file_to_view:
                    file_type = os.path.splitext(file_to_view)[1].lower()
                    if file_type == ".csv":
                        st.dataframe(pd.read_csv(file_to_view))
                    elif file_type == ".npy":
                        st.info("Numpy files contain binary data. You can load them in a script to analyze.")
                    elif file_type == ".pkl":
                        st.info("Pickle files contain model data and cannot be displayed directly.")
                    else:
                        st.info(f"File type {file_type} cannot be previewed here.")
        else:
            st.warning("`processed_data` directory not found.")

    # --- Models ---
    st.header("Models")
    models_path = os.path.join(ROOT_DIR, "models")
    if os.path.exists(models_path):
        model_files = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]
        st.write("Available models:")
        st.write(model_files)
    else:
        st.warning("`models` directory not found.")

    # --- Results ---
    st.header("Results")
    results_path = os.path.join(ROOT_DIR, "results")
    if os.path.exists(results_path):
        result_files = [f for f in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, f))]
        
        if result_files:
            selected_result = st.selectbox("Select a result file to view", result_files)
            if selected_result:
                result_path = os.path.join(results_path, selected_result)
                file_type = os.path.splitext(result_path)[1].lower()
                if file_type == ".csv":
                    st.dataframe(pd.read_csv(result_path))
                elif file_type == ".png":
                    st.image(result_path, caption=selected_result)
        else:
            st.info("No files found in the `results` directory.")
    else:
        st.warning("`results` directory not found.")
        
    # Check for missing files
    st.subheader("Missing Files Check")
    required_files = [
        os.path.join(ROOT_DIR, "all_predictions_combined.csv"),
        os.path.join(ROOT_DIR, "hybrid_folder_results.csv")
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
            
    if missing_files:
        st.warning("The following important files are missing:")
        for f in missing_files:
            st.write(f)
    else:
        st.success("All important files are present.")

def eda():
    '''Renders the Exploratory Data Analysis page.'''
    st.title("Exploratory Data Analysis")
    st.write("This section provides an overview of the dataset used for training the models.")

    OFFICIAL_DIR = os.path.join(ROOT_DIR, "proceed_data", "official")
    WIKI_DIR = os.path.join(ROOT_DIR, "proceed_data", "Wikipedia")
    FLATFIELD_DIR = os.path.join(ROOT_DIR, "proceed_data", "Flatfield")
    
    dataset_options = {
        "Official": OFFICIAL_DIR,
        "Wikipedia": WIKI_DIR,
        "Flatfield": FLATFIELD_DIR
    }

    dataset_choice = st.selectbox("Choose a dataset to analyze", list(dataset_options.keys()))

    if st.button("Run EDA"):
        with st.spinner(f"Running EDA on {dataset_choice} dataset..."):
            try:
                df, corrupted_files, duplicate_files = run_eda(dataset_options[dataset_choice])
                
                st.subheader("Dataset Overview")
                st.write(f"Total images in dataset: {len(df)}")
                st.write(f"Corrupted images: {len(corrupted_files)}")
                if corrupted_files:
                    for f in corrupted_files:
                        st.write(f"⚠️ {f}")
                else:
                    st.write("No corrupted images found!")

                st.write(f"Duplicate images: {len(duplicate_files)}")
                if duplicate_files:
                    for f in duplicate_files:
                        st.write(f"⚠️ {f}")
                else:
                    st.write("No duplicate images found!")

                if not df.empty:
                    st.write(f"Image heights: min={df['height'].min()}, max={df['height'].max()}, mean={df['height'].mean():.2f}")
                    st.write(f"Image widths: min={df['width'].min()}, max={df['width'].max()}, mean={df['width'].mean():.2f}")
                else:
                    st.info("No valid images found for statistics.")

                st.subheader("Data Table for " + dataset_choice)
                st.dataframe(df)

                st.subheader("Count by Scanner")
                fig, ax = plt.subplots()
                df['scanner'].value_counts().plot(kind='bar', ax=ax)
                ax.set_ylabel("Number of Images")
                st.pyplot(fig)

                st.subheader("Resolution Distribution")
                fig, ax = plt.subplots()
                df['resolution'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_ylabel("")
                st.pyplot(fig)

                st.subheader("Image Brightness Distribution")
                fig, ax = plt.subplots()
                ax.hist(df['brightness'], bins=30)
                ax.set_xlabel("Mean Pixel Intensity")
                ax.set_ylabel("Number of Images")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred during EDA: {e}")

def feature_extraction():
    '''Renders the Feature Extraction page.'''
    st.title("Feature Extraction")
    st.write("Run the feature extraction pipelines.")

    if st.button("Preprocess Images and Compute Residuals"):
        with st.spinner("Preprocessing images and computing residuals..."):
            preprocess_and_compute_residuals()
        st.success("Image preprocessing and residual computation complete.")

    if st.button("Compute Scanner Fingerprints"):
        with st.spinner("Computing scanner fingerprints..."):
            compute_scanner_fingerprints()
        st.success("Scanner fingerprint computation complete.")

    if st.button("Extract PRNU Features"):
        with st.spinner("Extracting PRNU features..."):
            extract_prnu_features()
        st.success("PRNU feature extraction complete.")

    if st.button("Extract Enhanced Features"):
        with st.spinner("Extracting enhanced features..."):
            extract_enhanced_features_func()
        st.success("Enhanced feature extraction complete.")

def testing():
    '''Renders the Testing page.'''
    st.title("Model Testing")
    st.write("Select a folder to run batch prediction on all images.")
    st.info("This page uses the **Hybrid CNN model** for batch prediction.")

    default_test_path = os.path.join(ROOT_DIR, "proceed_data", "Test")
    folder_path = st.text_input("Enter the path to the folder containing images:", default_test_path)

    if st.button("Run Batch Prediction"):
        if os.path.isdir(folder_path):
            with st.spinner(f"Running batch prediction on folder: {folder_path}"):
                try:
                    results = predict_folder(folder_path, output_csv="hybrid_folder_results.csv")
                    st.success("Batch prediction complete!")
                    st.dataframe(pd.DataFrame(results, columns=["Image", "Predicted Label", "Confidence (%)"]))
                except Exception as e:
                    st.error(f"An error occurred during batch prediction: {e}")
        else:
            st.error("The specified path is not a valid directory.")

def cnn_model():
    '''Renders the CNN Model page.'''
    st.title("CNN Model Management")
    st.write("Train and evaluate the hybrid CNN model.")

    if st.button("Train Hybrid CNN Model"):
        with st.spinner("Training Hybrid CNN model..."):
            try:
                python_executable = sys.executable
                train_script_path = os.path.join(ROOT_DIR, "src", "cnn_model", "train_hybrid_cnn.py")
                
                st.write("Starting training process...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                process = subprocess.Popen(
                    [python_executable, train_script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='replace'
                )

                output_lines = []
                total_epochs = 50  # Default, will be updated

                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line)
                    # Show the last few lines of output
                    status_text.code("".join(output_lines[-10:]))

                    match = re.search(r"Epoch (\d+)/(\d+)", line)
                    if match:
                        epoch = int(match.group(1))
                        total_epochs = int(match.group(2))
                        progress_bar.progress(epoch / total_epochs)
                
                process.wait()
                progress_bar.progress(1.0)

                if process.returncode == 0:
                    st.success("Model training complete!")
                    history_path = os.path.join(ROOT_DIR, "proceed_data", "hybrid_training_history.pkl")
                    if os.path.exists(history_path):
                        with open(history_path, "rb") as f:
                            history = pickle.load(f)
                        st.write("Training History:")
                        df_history = pd.DataFrame(history)
                        st.line_chart(df_history)
                else:
                    st.error("Model training failed.")
                    st.code("".join(output_lines))

            except Exception as e:
                st.error(f"An error occurred during training: {e}")

    if st.button("Evaluate Hybrid CNN Model"):
        with st.spinner("Evaluating Hybrid CNN model..."):
            try:
                python_executable = sys.executable
                eval_script_path = os.path.join(ROOT_DIR, "src", "cnn_model", "eval_hybrid_cnn.py")

                # Run the evaluation script
                result = subprocess.run([python_executable, eval_script_path], capture_output=True, text=True)

                st.success("Model evaluation complete!")
                st.write("---" + "Evaluation Results" + "---")

                # --- Read results from the generated CSV file ---
                report_path = os.path.join(ROOT_DIR, "results", "classification_report.csv")
                if os.path.exists(report_path):
                    try:
                        df_report = pd.read_csv(report_path, index_col=0)

                        # --- Accuracy (from CSV) ---
                        if 'accuracy' in df_report.index:
                            # The accuracy value is typically in the first column for the 'accuracy' row
                            accuracy_value = df_report.loc['accuracy'].iloc[0]
                            st.metric("Test Accuracy", f"{accuracy_value*100:.2f}%")
                            # Drop the accuracy row for a cleaner report display
                            df_display = df_report.drop('accuracy')
                        else:
                            st.warning("Accuracy not found in the classification report file.")
                            df_display = df_report

                        # --- Classification Report (from CSV) ---
                        st.subheader("Classification Report")
                        st.dataframe(df_display)

                    except Exception as e:
                        st.error(f"Error reading or processing classification report file: {e}")
                else:
                    st.warning("Classification report file not found. Raw output below:")
                    st.code(result.stdout) # Show raw output if CSV is not found

                # --- Errors from script ---
                if result.stderr:
                    st.write("Errors:")
                    st.code(result.stderr)

                # --- Confusion Matrix ---
                conf_matrix_path = os.path.join(ROOT_DIR, "results", "CNN_confusion_matrix.png")
                if os.path.exists(conf_matrix_path):
                    st.image(conf_matrix_path, caption='Confusion Matrix', width='stretch')
                else:
                    st.warning("Confusion matrix image not found.")

            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")

def model_performance():
    st.title("Model Performance")

    # Define paths to the result files
    results_dir = os.path.join(ROOT_DIR, "results")
    cnn_report_path = os.path.join(results_dir, "classification_report.csv")
    cnn_matrix_path = os.path.join(results_dir, "CNN_confusion_matrix.png")
    rf_report_path = os.path.join(results_dir, "Random_Forest_classification_report.csv")
    rf_matrix_path = os.path.join(results_dir, "Random_Forest_confusion_matrix.png")
    svm_report_path = os.path.join(results_dir, "SVM_classification_report.csv")
    svm_matrix_path = os.path.join(results_dir, "SVM_confusion_matrix.png")
    hybrid_matrix_path = os.path.join(results_dir, "hybrid_confusion_matrix_from_csv.png")
    history_path = os.path.join(ROOT_DIR, "proceed_data", "hybrid_training_history.pkl")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CNN", "Random Forest", "SVM", "Hybrid", "Training Curves"])

    with tab1:
        st.header("CNN Model")
        if os.path.exists(cnn_report_path):
            st.subheader("Classification Report")
            df_cnn = pd.read_csv(cnn_report_path)
            st.dataframe(df_cnn)
        else:
            st.warning("CNN classification report not found.")
        
        if os.path.exists(cnn_matrix_path):
            st.subheader("Confusion Matrix")
            st.image(cnn_matrix_path, caption='CNN Confusion Matrix')
        else:
            st.warning("CNN confusion matrix not found.")

    with tab2:
        st.header("Random Forest Model")
        if os.path.exists(rf_report_path):
            st.subheader("Classification Report")
            df_rf = pd.read_csv(rf_report_path)
            st.dataframe(df_rf)
        else:
            st.warning("Random Forest classification report not found.")

        if os.path.exists(rf_matrix_path):
            st.subheader("Confusion Matrix")
            st.image(rf_matrix_path, caption='Random Forest Confusion Matrix')
        else:
            st.warning("Random Forest confusion matrix not found.")

    with tab3:
        st.header("SVM Model")
        if os.path.exists(svm_report_path):
            st.subheader("Classification Report")
            df_svm = pd.read_csv(svm_report_path)
            st.dataframe(df_svm)
        else:
            st.warning("SVM classification report not found.")

        if os.path.exists(svm_matrix_path):
            st.subheader("Confusion Matrix")
            st.image(svm_matrix_path, caption='SVM Confusion Matrix')
        else:
            st.warning("SVM confusion matrix not found.")
            
    with tab4:
        st.header("Hybrid Model")
        if os.path.exists(hybrid_matrix_path):
            st.subheader("Confusion Matrix")
            st.image(hybrid_matrix_path, caption='Hybrid Confusion Matrix')
        else:
            st.warning("Hybrid confusion matrix not found.")

    with tab5:
        st.header("Training Curves")
        if os.path.exists(history_path):
            if st.button("Show Training Curves"):
                with open(history_path, "rb") as f:
                    history = pickle.load(f)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

                # Accuracy
                ax1.plot(history['accuracy'], label='Train Accuracy')
                ax1.plot(history['val_accuracy'], label='Validation Accuracy')
                ax1.set_title('Model Accuracy')
                ax1.set_ylabel('Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.legend(loc='upper left')

                # Loss
                ax2.plot(history['loss'], label='Train Loss')
                ax2.plot(history['val_loss'], label='Validation Loss')
                ax2.set_title('Model Loss')
                ax2.set_ylabel('Loss')
                ax2.set_xlabel('Epoch')
                ax2.legend(loc='upper left')

                st.pyplot(fig)
        else:
            st.warning("Training history not found.")

def overall_performance_and_baseline_models():
    st.title("Overall Performance & Baseline Models")

    st.header("Performance Visualization")

    if st.button("Show Combined Predictions"):
        try:
            df = pd.read_csv(os.path.join(ROOT_DIR, "all_predictions_combined.csv"))
            st.dataframe(df)
            
            st.subheader("Predicted Scanner Distribution")
            fig, ax = plt.subplots()
            df['predicted_scanner'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        except FileNotFoundError:
            st.error("all_predictions_combined.csv not found.")

    if st.button("Show Hybrid Folder Results"):
        try:
            df = pd.read_csv(os.path.join(ROOT_DIR, "hybrid_folder_results.csv"))
            st.dataframe(df)

            st.subheader("Predicted Scanner Distribution (Hybrid)")
            fig, ax = plt.subplots()
            df['Predicted_Label'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        except FileNotFoundError:
            st.error("hybrid_folder_results.csv not found.")

    st.header("Baseline Models (Random Forest & SVM)")

    if st.button("Train Baseline Models"):
        with st.spinner("Training Random Forest and SVM models..."):
            try:
                python_executable = sys.executable
                train_script_path = os.path.join(ROOT_DIR, "src", "baseline", "train_baseline.py")
                
                result = subprocess.run([python_executable, train_script_path], capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("Baseline models trained successfully!")
                    st.code(result.stdout)
                else:
                    st.error("Baseline model training failed.")
                    st.code(result.stderr)

            except Exception as e:
                st.error(f"An error occurred during training: {e}")

    if st.button("Evaluate Baseline Models"):
        with st.spinner("Evaluating baseline models..."):
            try:
                python_executable = sys.executable
                eval_script_path = os.path.join(ROOT_DIR, "src", "baseline", "evaluate_baseline.py")
                
                result = subprocess.run([python_executable, eval_script_path], capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("Baseline model evaluation complete!")
                    st.code(result.stdout)

                    rf_report_path = os.path.join(ROOT_DIR, "results", "Random_Forest_classification_report.csv")
                    svm_report_path = os.path.join(ROOT_DIR, "results", "SVM_classification_report.csv")
                    rf_matrix_path = os.path.join(ROOT_DIR, "results", "Random_Forest_confusion_matrix.png")
                    svm_matrix_path = os.path.join(ROOT_DIR, "results", "SVM_confusion_matrix.png")

                    st.subheader("Random Forest Classification Report")
                    if os.path.exists(rf_report_path):
                        df_rf = pd.read_csv(rf_report_path)
                        st.dataframe(df_rf)
                    else:
                        st.warning("Random Forest classification report not found.")

                    st.subheader("Random Forest Confusion Matrix")
                    if os.path.exists(rf_matrix_path):
                        st.image(rf_matrix_path, caption='Random Forest Confusion Matrix')
                    else:
                        st.warning("Random Forest confusion matrix not found.")

                    st.subheader("SVM Classification Report")
                    if os.path.exists(svm_report_path):
                        df_svm = pd.read_csv(svm_report_path)
                        st.dataframe(df_svm)
                    else:
                        st.warning("SVM classification report not found.")

                    st.subheader("SVM Confusion Matrix")
                    if os.path.exists(svm_matrix_path):
                        st.image(svm_matrix_path, caption='SVM Confusion Matrix')
                    else:
                        st.warning("SVM confusion matrix not found.")
                else:
                    st.error("Baseline model evaluation failed.")
                    st.code(result.stderr)

            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")

    st.subheader("Predict with Baseline Model")
    baseline_model_choice = st.selectbox("Choose a baseline model", ["Random Forest", "SVM"])
    baseline_uploaded_file = st.file_uploader("Choose an image for baseline prediction...", type=["jpg", "png", "tif"], key="baseline_uploader")

    if baseline_uploaded_file is not None:
        st.image(baseline_uploaded_file, caption='Uploaded Image.', width='stretch')
        if st.button("Predict with Baseline Model"):
            with st.spinner(f'Analyzing and predicting with {baseline_model_choice}...'):
                try:
                    temp_path = os.path.join(ROOT_DIR, "temp_image_baseline.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(baseline_uploaded_file.getbuffer())
                    
                    model_choice_short = "rf" if baseline_model_choice == "Random Forest" else "svm"
                    pred, conf, prob, classes = predict_with_baseline_model(temp_path, model_choice=model_choice_short)

                    st.success(f"Predicted Scanner Model: **{pred}**")
                    st.metric(label="Confidence", value=f"{conf:.2f}%")
                    st.info(f"Prediction made using: **{baseline_model_choice}**")

                    prob_df = pd.DataFrame({
                        'Scanner Model': classes,
                        'Probability': prob
                    })
                    st.dataframe(prob_df)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

def about():
    """Renders the About page with report generation."""
    st.title("About AI TraceFinder")
    st.markdown("""
    **AI TraceFinder** is a powerful tool designed for digital image forensics. It specializes in identifying the source scanner of a document by analyzing residual patterns and noise inherent in the scanning process. This application provides a suite of utilities for forensic analysis, including model training, evaluation, and batch processing of images.

    ### Key Features:
    - **Scanner Identification:** Predict the source scanner of an image using advanced machine learning models.
    - **Multiple Model Support:** Utilizes a Hybrid CNN, Random Forest, and SVM models for robust analysis.
    - **Data Analysis:** Offers Exploratory Data Analysis (EDA) to understand the underlying data distributions.
    - **Feature Extraction:** Implements various techniques to extract features like PRNU, LBP, and FFT-based features.
    - **Model Evaluation:** Provides detailed performance metrics and visualizations for each model.

    ### Technologies Used:
    - **Backend:** Python, TensorFlow, scikit-learn
    - **Frontend:** Streamlit
    - **Libraries:** OpenCV, PyWavelets, NumPy, Pandas, Matplotlib

    This project is intended for educational and research purposes. The models have been trained on a specific dataset, and performance may vary on other data.
    
    ---

    **Team:** RUSHIKESH RAGHATATE | **Institution:** GH RAISONI COLLEGE OF ENGINEERING , NAGPUR

    **GitHub Repo:** [https://github.com/rushikeshraghatate90/AI_TraceFinder](https://github.com/rushikeshraghatate90/AI_TraceFinder)
    """)

    st.markdown("---")
    st.header("Generate Performance Report")
    st.write("Click the button below to generate a comprehensive PDF report of the model performances.")

    try:
        from fpdf import FPDF

        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                results_dir = os.path.join(ROOT_DIR, "results")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "AI TraceFinder - Model Performance Report", 0, 1, 'C')
                pdf.ln(10)

                def add_report_section(pdf, model_name, report_path, matrix_path):
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, f"{model_name} Model", 0, 1)
                    
                    if os.path.exists(report_path):
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "Classification Report:", 0, 1)
                        pdf.set_font("Courier", size=8)
                        
                        df = pd.read_csv(report_path)
                        
                        # Headers
                        header = list(df.columns)
                        pdf.set_fill_color(200, 220, 255)
                        pdf.cell(40, 6, header[0], 1, 0, 'C', 1)
                        for col in header[1:]:
                            pdf.cell(30, 6, col, 1, 0, 'C', 1)
                        pdf.ln()

                        # Data
                        pdf.set_fill_color(255, 255, 255)
                        for index, row in df.iterrows():
                            pdf.cell(40, 6, str(row.iloc[0]), 1, 0, 'L', 1)
                            for item in row.iloc[1:]:
                                pdf.cell(30, 6, f"{item:.2f}", 1, 0, 'C', 1)
                            pdf.ln()
                    else:
                        pdf.set_font("Arial", size=10)
                        pdf.cell(0, 10, "Classification report not found.", 0, 1)

                    if os.path.exists(matrix_path):
                        pdf.ln(5)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "Confusion Matrix:", 0, 1)
                        pdf.image(matrix_path, w=150)
                    else:
                        pdf.set_font("Arial", size=10)
                        pdf.cell(0, 10, "Confusion matrix not found.", 0, 1)
                    pdf.ln(5)

                # --- Add sections for each model ---
                add_report_section(pdf, "CNN", 
                                   os.path.join(results_dir, "classification_report.csv"),
                                   os.path.join(results_dir, "CNN_confusion_matrix.png"))
                
                pdf.add_page()
                add_report_section(pdf, "Random Forest",
                                   os.path.join(results_dir, "Random_Forest_classification_report.csv"),
                                   os.path.join(results_dir, "Random_Forest_confusion_matrix.png"))

                pdf.add_page()
                add_report_section(pdf, "SVM",
                                   os.path.join(results_dir, "SVM_classification_report.csv"),
                                   os.path.join(results_dir, "SVM_confusion_matrix.png"))

                hybrid_matrix_path = os.path.join(results_dir, "hybrid_confusion_matrix_from_csv.png")
                if os.path.exists(hybrid_matrix_path):
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 10, "Hybrid Model", 0, 1)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Confusion Matrix:", 0, 1)
                    pdf.image(hybrid_matrix_path, w=150)

                # --- Save and Download PDF ---
                pdf_output_path = os.path.join(ROOT_DIR, "AI_TraceFinder_Report.pdf")
                pdf.output(pdf_output_path)
                
                with open(pdf_output_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name="AI_TraceFinder_Report.pdf",
                        mime="application/pdf"
                    )
                st.success("Report generated successfully!")

    except ImportError:
        st.error("PDF generation requires the `fpdf2` library. Please install it by running: `pip install fpdf2`")
    except Exception as e:
        st.error(f"An error occurred while generating the report: {e}")

def main():
    '''Main function to run the Streamlit app.'''
    set_app_style()
    st.sidebar.title("AI TraceFinder")
    st.sidebar.markdown("Digital Scanner Forensics")

    pages = {
        "Home": home,
        "Prediction": prediction,
        "Folder Analysis": folder_analysis,
        "EDA": eda,
        "Feature Extraction": feature_extraction,
        "Testing": testing,
        "CNN Model": cnn_model,
        "Model Performance": model_performance,
        "Overall Performance & Baseline Models": overall_performance_and_baseline_models,
        "About": about,
    }

    selection = st.sidebar.radio("Navigation", list(pages.keys()))

    # Render the selected page
    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
