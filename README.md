# Speech Emotion Recognition

**ML II Final Project — University of Chicago**

**Authors:** Lam Tran, Kevin Wang, Jason Clark, Yining Mao

---

## Overview

Many real-world systems — virtual assistants, customer service bots, healthcare monitoring tools — cannot naturally understand human emotions during speech interactions. This project builds a speech emotion recognition (SER) system that predicts emotions directly from audio recordings using the RAVDESS dataset.

---

## Dataset

**RAVDESS Emotional Speech Audio**
- Source: [Kaggle — uwrfkaggler/ravdess-emotional-speech-audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- License: CC BY-NC-SA 4.0
- 1,440 `.wav` clips across 24 professional actors (12 male, 12 female)
- 48,000 Hz native sampling rate; mean clip duration ≈ 3.7s

**8 emotion classes:**

| Code | Label     |
|------|-----------|
| 01   | neutral   |
| 02   | calm      |
| 03   | happy     |
| 04   | sad       |
| 05   | angry     |
| 06   | fearful   |
| 07   | disgust   |
| 08   | surprised |

Neutral has ~96 samples (half the others) because RAVDESS omits the "strong" intensity variant for neutral. All other classes have 192 samples each. Every actor contributes exactly 60 recordings.

---

## Project Structure

```
.
├── ml2_final_project.ipynb   # Main notebook (run in Google Colab)
└── README.md
```

---

## Pipeline

### 1. Data ingestion
- Authenticates with Kaggle API via a `kaggle.json` token stored in Google Drive
- Downloads and extracts the RAVDESS zip into `/content/data/ravdess/`
- Deduplicates the two parallel folder trees the zip creates

### 2. Label parsing
Emotion labels are encoded in filenames (e.g. `03-01-05-01-02-01-12.wav` → position 3 = emotion code). No separate CSV is needed.

### 3. Data cleaning & validation
All checks passed on the clean RAVDESS dataset:
- All files are `.wav` and follow the 7-part naming convention
- No invalid emotion codes, missing labels, or duplicate paths
- Every file loads cleanly through librosa
- Every actor has exactly 60 recordings
- No suspiciously silent clips (RMS < 1e-3)

### 4. Exploratory data analysis
- Class distribution (slight imbalance on neutral)
- Per-actor counts (perfectly balanced)
- Clip duration distribution (mean ≈ 3.7s, std ≈ 0.34s)
- Waveform, spectrogram, and MFCC visualizations per emotion

### 5. Feature extraction
Each clip is converted to a **40-dimensional MFCC feature vector** by:
1. Resampling to 22,050 Hz
2. Computing 40 MFCC coefficients over time frames → shape `(40, n_frames)`
3. Averaging across the time axis → shape `(40,)`

This fixed-length representation lets traditional ML models consume audio directly.

### 6. Preprocessing
- Labels encoded with `LabelEncoder` (strings → integers 0–7)
- Stratified 80/20 train/test split (1,152 train / 288 test)
- Features standardized with `StandardScaler` (fit on train only)

---

## Models & Results

| Model | Test Accuracy | Macro F1 |
|-------|:---:|:---:|
| Logistic Regression | 42.7% | 0.409 |
| **Random Forest (baseline)** | **58.7%** | **0.547** |
| Dense Neural Network (MLP) | 56.6% | 0.540 |

**Random Forest** (300 trees) outperforms Logistic Regression by ~16 points and narrowly beats the MLP. It serves as the official baseline.

**MLP architecture:**
```
Input(40) → Dense(128) + BN + ReLU + Dropout(0.4)
           → Dense(64)  + BN + ReLU + Dropout(0.4)
           → Dense(8, softmax)
```
Trained with Adam (lr=1e-3), L2 regularization, and early stopping (patience=10).

**Hardest emotions to classify:** neutral (F1 ≈ 0.22–0.24) and happy (F1 ≈ 0.46), likely due to class imbalance for neutral and overlap with other positive-valence emotions for happy.

---

## Getting Started

### Kaggle credentials (required)

The notebook downloads the RAVDESS dataset automatically via the Kaggle API. You need a `kaggle.json` credentials file before running anything.

1. Sign in at [kaggle.com](https://www.kaggle.com) (free account)
2. Go to **Settings → API → Create New Token** — this downloads `kaggle.json`
3. Place it in the standard location for your environment:

**Google Colab:** upload `kaggle.json` to your Google Drive at `MyDrive/ml2/kaggle.json`. The notebook will mount Drive and copy it automatically.

**Local:** place it at `~/.kaggle/kaggle.json` and lock down permissions:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

The notebook detects which environment it's running in and handles credentials accordingly. If the file is missing locally, it will raise a clear error with instructions.

### Running the notebook

**Google Colab**
1. Open `ml2_final_project.ipynb` in Colab
2. Run all cells top to bottom — Drive will mount, RAVDESS will download, features will extract, and all models will train

**Local**
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Launch Jupyter and run all cells:
```bash
jupyter notebook ml2_final_project.ipynb
```
The dataset (~430 MB) will be downloaded to `./data/` on first run.

---

## Notes

- **Speaker leakage:** the current split is random, so the same actor can appear in both train and test. A speaker-independent split (hold out actors entirely) would give a more honest estimate of generalization.
- **Feature choice:** time-averaged MFCCs discard temporal dynamics. Future work could use MFCC sequences fed into an LSTM or CNN-LSTM to exploit how emotions evolve over time.
- **Class imbalance:** macro F1 is reported alongside accuracy to avoid neutral's smaller class being ignored.
