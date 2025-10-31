# 📊 AI TraceFinder: Forensic Scanner Identification

Detecting document forgery by analyzing a scanner's unique digital fingerprint.

---

## 📘 Table of Contents
- [About the Project](#-about-the-project)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Accuracy & Performance](#-accuracy--performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 About the Project

Scanned documents like legal certificates, IDs, and agreements can easily be forged using unauthorized scanners.  
However, every scanner introduces a **unique digital fingerprint** in each image — microscopic noise, texture variations, and compression patterns.  

**AI TraceFinder** identifies these subtle traces to detect which scanner was used to produce a scanned image — helping to **authenticate documents** and detect **digital forgery**.

### 🧠 Objectives
- Identify the source scanner from a digital image.  
- Detect tampered or fake documents.  
- Assist in forensic and legal document verification.

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|-----------|-------------|----------|
| **Backend / ML** | Python | Core programming language |
| | Scikit-learn | Random Forest & SVM Baseline Models |
| | TensorFlow / Keras | Deep Learning (CNN / Hybrid CNN) |
| | OpenCV | Image processing and feature extraction |
| | NumPy, Pandas | Data operations and feature engineering |
| **Frontend / Visualization** | Streamlit | Interactive web dashboard |
| | Matplotlib, Seaborn | Visualizations and plots |
| | Pillow (PIL) | Image display in Streamlit |
| **Utilities** | Git & GitHub | Version control |
| | venv | Virtual environment setup |

---

## ✨ Features

- 🧩 **Feature Extraction Module:** Extracts statistical and texture-based residuals for ML training.  
- 🤖 **Baseline Models:** Trains and evaluates SVM & Random Forest models.  
- 🧠 **Deep Learning (CNN):** Dual-branch CNN & Hybrid CNN architectures for raw image classification.  
- 📊 **Data Visualization:** Display class distribution, confusion matrices, and metrics.  
- 💾 **Result Exporting:** Automatically saves results and reports in `/results`.  
- 🌐 **Streamlit Interface:** Simple upload → predict → get scanner source instantly.

---

## 📈 Accuracy & Performance

| Model | Accuracy | Precision | Recall | F1-score | Test Samples |
|--------|-----------|------------|----------|-----------|---------------|
| **Hybrid CNN** | **92.21%** | 0.93 | 0.92 | 0.92 | 517 images |

---

## 🚀 Installation

### 1️⃣ Clone or Download the Repository
```bash
git clone https://github.com/rushikeshraghatate90/AI_TraceFinder
cd AI-TraceFinder
```

## ⚙️ Setup & Usage Guide (Quick Start)

```bash
# 📦 If downloaded as ZIP, extract to:
C:\Users\ASUS\Downloads\AI
```
```
# 🧰 1️⃣ Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate
```
```
# 📦 2️⃣ Install Dependencies
pip install -r requirements.txt
```
```
# 🧠 3️⃣ Run the Streamlit App (from project root)
streamlit run scr/app/main_app.py
```

## 🚀 Inside the App:

#### 📤 Upload a scanned image.
#### 🧮 Select model — Baseline (RF/SVM) or CNN.
#### 📈 View predicted scanner and confidence score.
#### 💾 Access evaluation metrics and visualizations.


## 🗂 Project Structure
```
C:\Users\ASUS\Downloads\AI
│
├── Data/                              # Raw scanned image datasets
│   ├── Flatfield/
│   ├── Official/
│   ├── Tampered images/
│   └── wikipedia/
│       ├── enhanced_features.pkl
│       ├── flatfield_residuals.pkl
│       ├── hybrid_feat_scaler.pkl
│       ├── hybrid_label_encoder.pkl
│       └── official_datasets.pkl
│
├── logs/                              # Model training & evaluation logs
│
├── models/                            # Trained models
│   ├── dual_branch_cnn.h5
│   ├── hybrid_model.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   └── scaler.pkl
│
├── results/                           # Evaluation reports and plots
│
├── scr/
│   ├── app/
│   │   └── main_app.py                # Streamlit main interface
│   ├── baseline/
│   │   ├── predict_baseline.py
│   │   └── train_baseline.py
│   ├── cnn/
│   │   ├── predict_cnn.py
│   │   └── train_cnn.py
│   ├── hybrid_cnn/
│   │   ├── eval_hybrid_cnn.py
│   │   └── visualize_data.py
│   └── eda.py                         # Exploratory analysis script
│
├── processed_data/                    # Preprocessed intermediate data
├── residuals/                         # Extracted noise & residual maps
├── Residuals_Paths/                   # Path references for residuals
├── requirements.txt
└── README.md
```

## 🤝 Contributing

- git checkout -b feature/your-feature
- git commit -m "Add your feature"
- git push origin feature/your-feature
#### Then open a Pull Request.


## 📜 License

#### This project is open-source under the MIT License.
#### See the LICENSE file for details.


## 📬 Contact

##### 👤 Author: Raghatate Rushikesh
##### 📧 Email: rushikeshraghatate90@gmail.com
##### 💼 GitHub: https://github.com/rushikeshraghatate90
