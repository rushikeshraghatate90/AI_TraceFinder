"""
EDA for Image Dataset (Local)
Dataset structure:
- Official, Wikipedia, Flatfield
- Each folder contains subfolders per class or scanner
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
from tabulate import tabulate

# ---------------------------
# 1) Dataset Paths
# ---------------------------
OFFICIAL_DIR = "Data/Official"
WIKI_DIR = "Data/Wikipedia"
FLATFIELD_DIR = "Data/Flatfield"

EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# ---------------------------
# 2) Utility
# ---------------------------
def is_image_file(fname):
    return os.path.splitext(fname)[1].lower() in EXTENSIONS


# ---------------------------
# 3) EDA Function
# ---------------------------
def run_eda(dataset_path):
    print(f"\n=== EDA for {dataset_path} ===")

    if not os.path.exists(dataset_path):
        print(f"❌ Path does not exist: {dataset_path}")
        return

    class_counts = {}
    total_images = 0
    corrupted_files = []
    image_shapes = []
    brightness_values = []
    duplicates = []

    hashes = {}

    # Walk through dataset
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        subfolder_counts = {}
        class_total = 0

        for root, dirs, files in os.walk(class_path):
            sub_count = 0
            for f in files:
                if not is_image_file(f):
                    continue

                file_path = os.path.join(root, f)

                # Check if file is corrupted
                img = cv2.imread(file_path)
                if img is None:
                    corrupted_files.append(file_path)
                    continue

                # Image stats
                h, w = img.shape[:2]
                image_shapes.append((h, w))
                brightness_values.append(float(img.mean()))

                # Check duplicates via hash
                try:
                    with open(file_path, "rb") as f_img:
                        img_hash = hashlib.md5(f_img.read()).hexdigest()
                    if img_hash in hashes:
                        duplicates.append(file_path)
                    else:
                        hashes[img_hash] = file_path
                except Exception as e:
                    print(f"⚠️ Error hashing file: {file_path} ({e})")

                sub_count += 1
                class_total += 1

            subfolder_name = os.path.relpath(root, dataset_path)
            subfolder_counts[subfolder_name] = sub_count

        # Table per class
        print(f"\nClass: {class_name}")
        table = [[sub, cnt] for sub, cnt in subfolder_counts.items()]
        if table:
            print(tabulate(table, headers=["Subfolder", "Number of Images"], tablefmt="grid"))
        print(f"Total images in class '{class_name}': {class_total}\n")

        class_counts[class_name] = class_total
        total_images += class_total

    # ---------------------------
    # Summary
    # ---------------------------
    print(f"Total images in dataset: {total_images}")
    print(f"Corrupted images: {len(corrupted_files)}")
    if corrupted_files:
        for f in corrupted_files[:10]:
            print(f"⚠️ {f}")
        if len(corrupted_files) > 10:
            print(f"...and {len(corrupted_files) - 10} more")

    else:
        print("✅ No corrupted images found!")

    print(f"Duplicate images: {len(duplicates)}")
    if duplicates:
        for f in duplicates[:10]:
            print(f"⚠️ {f}")
        if len(duplicates) > 10:
            print(f"...and {len(duplicates) - 10} more")
    else:
        print("✅ No duplicate images found!")

    # ---------------------------
    # Plots
    # ---------------------------
    if class_counts:
        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
        plt.xticks(rotation=45)
        plt.ylabel("Number of Images")
        plt.title(f"Class Distribution - {os.path.basename(dataset_path)}")
        plt.tight_layout()
        plt.show()

    if image_shapes:
        heights, widths = zip(*image_shapes)
        print(f"\nImage heights: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.2f}")
        print(f"Image widths: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.2f}")

        # Aspect ratio
        aspect_ratios = [w / h for h, w in image_shapes if h > 0]
        plt.hist(aspect_ratios, bins=20, color="lightgreen", edgecolor="black")
        plt.xlabel("Width / Height")
        plt.ylabel("Number of Images")
        plt.title(f"Aspect Ratio Distribution - {os.path.basename(dataset_path)}")
        plt.show()

        # Brightness
        plt.hist(brightness_values, bins=30, color="orange", edgecolor="black")
        plt.xlabel("Mean Pixel Intensity")
        plt.ylabel("Number of Images")
        plt.title(f"Brightness Distribution - {os.path.basename(dataset_path)}")
        plt.show()

    # ---------------------------
    # Random sample previews
    # ---------------------------
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        all_images = []
        for root, dirs, files in os.walk(class_path):
            for f in files:
                if is_image_file(f):
                    all_images.append(os.path.join(root, f))

        if not all_images:
            continue

        sample_files = random.sample(all_images, min(3, len(all_images)))

        plt.figure(figsize=(10, 3))
        shown = 0
        for i, fpath in enumerate(sample_files):
            img = cv2.imread(fpath)
            if img is None:
                print(f"⚠️ Skipping unreadable sample: {fpath}")
                continue
            plt.subplot(1, len(sample_files), shown + 1)
            try:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"⚠️ Error displaying {fpath}: {e}")
                continue
            plt.title(os.path.basename(fpath), fontsize=8)
            plt.axis("off")
            shown += 1

        if shown > 0:
            plt.suptitle(f"Random Samples - {class_name}")
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ No valid samples to display for {class_name}")


# ---------------------------
# 4) Run EDA on all datasets
# ---------------------------
if __name__ == "__main__":
    for path in [OFFICIAL_DIR, WIKI_DIR, FLATFIELD_DIR]:
        run_eda(path)

