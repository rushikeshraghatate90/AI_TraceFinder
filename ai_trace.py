import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="AI Forensic Metadata Extractor", layout="wide")
st.title("📂 AI Forensic Dataset Metadata Extractor")

def extract_metadata(image_path, class_label):
    """Extract simple image metadata."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        file_size = os.path.getsize(image_path) / 1024  # KB
        resolution = f"{width} x {height}"

        return {
            "file_name": os.path.basename(image_path),
            "class": class_label,
            "width": width,
            "height": height,
            "file_size_kb": round(file_size, 2),
            "pixel_resolution": resolution,
            "error": None
        }
    except Exception as e:
        return {
            "file_name": os.path.basename(image_path),
            "class": class_label,
            "width": None,
            "height": None,
            "file_size_kb": None,
            "pixel_resolution": None,
            "error": str(e)
        }

dataset_root = st.text_input("📂 Enter dataset root path:", "")

if dataset_root and os.path.isdir(dataset_root):
    st.info("🔎 Scanning dataset...")
    records = []

    # Decide structure based on root path
    nested_structure_root = r"C:\Users\ASUS\OneDrive\Desktop\FOOTBALL_ANALYSIS\AI_TraceFinder\Official"
    
    folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    st.success(f"Detected {len(folders)} folders: {folders}")

    for folder in folders:
        folder_path = os.path.join(dataset_root, folder)

        if dataset_root == nested_structure_root:
            # Nested dataset/device structure
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            if subfolders:
                for sub in subfolders:
                    sub_path = os.path.join(folder_path, sub)
                    files = [f for f in os.listdir(sub_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    st.write(f"Folder '{folder}/{sub}' → {len(files)} images")
                    for fname in files:
                        rec = extract_metadata(os.path.join(sub_path, fname), f"{folder}_{sub}")
                        records.append(rec)
        else:
            # Flat structure
            files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            st.write(f"Folder '{folder}' → {len(files)} images")
            for fname in files:
                rec = extract_metadata(os.path.join(folder_path, fname), folder)
                records.append(rec)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    st.subheader("📊 Metadata Extracted (Preview)")
    st.dataframe(df.head(20))

    # Save CSV
    save_path = os.path.join(dataset_root, "forensic_metadata.csv")
    df.to_csv(save_path, index=False)
    st.success(f"💾 Metadata saved to {save_path}")

    # Class distribution
    if "class" in df.columns:
        st.subheader("📌 Class Distribution")
        st.bar_chart(df["class"].value_counts())

    # Sample images
    st.subheader("🖼 Sample Images")
    cols = st.columns(5)
    for idx, cls in enumerate(df["class"].unique()):
        # handle nested or flat
        parts = cls.split("_")
        if dataset_root == nested_structure_root and len(parts) == 2:
            sample_img_path = os.path.join(dataset_root, parts[0], parts[1], df[df["class"]==cls].iloc[0]["file_name"])
        else:
            sample_img_path = os.path.join(dataset_root, cls, df[df["class"]==cls].iloc[0]["file_name"])
        try:
            img = Image.open(sample_img_path)
            cols[idx % 5].image(img, caption=cls, use_container_width=True)
        except:
            pass

elif dataset_root:
    st.error("❌ Invalid dataset path. Please enter a valid folder.")
