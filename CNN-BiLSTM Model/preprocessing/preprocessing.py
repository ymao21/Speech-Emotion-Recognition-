"""
Feature extraction pipeline for RAVDESS emotion recognition.

This script:
1. Recursively loads .wav files from a dataset directory
2. Parses RAVDESS emotion metadata from filenames
3. Extracts frame-level audio features over time
4. Standard-scales all acoustic features (fit on train, transform both splits)
5. Returns train/test DataFrames ready for ML workflows
6. Optionally exports the features to Parquet

Author: Lam Tran
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "data/raw"

# Frame extraction parameters
SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 40

# Columns that are metadata / targets — excluded from scaling
METADATA_COLS = [
    "file_path",
    "emotion_code",
    "emotion_label",
    "actor_id",
    "frame_index",
    "time_seconds",
]


# ============================================================================
# LABEL PARSING
# ============================================================================

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_filename(filepath: str) -> dict[str, str] | None:
    """
    Parse metadata from a RAVDESS filename.

    Example filename:
        03-01-05-01-02-02-12.wav

    Returns:
        Dictionary containing:
            - emotion_code
            - emotion_label
            - actor_id

        Returns None if filename format is invalid.
    """
    name = Path(filepath).stem
    parts = name.split("-")

    if len(parts) != 7:
        return None

    emotion_code = parts[2]
    actor_id = parts[6]
    emotion_label = EMOTION_MAP.get(emotion_code)

    if emotion_label is None:
        return None

    return {
        "emotion_code": emotion_code,
        "emotion_label": emotion_label,
        "actor_id": actor_id,
    }


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_frame_features(
    audio: np.ndarray,
    sample_rate: int,
) -> pd.DataFrame:
    """
    Extract frame-level acoustic features from audio.

    Features extracted:
        - MFCCs (40)
        - Delta MFCCs
        - Delta-Delta MFCCs
        - RMS Energy
        - Zero Crossing Rate
        - Spectral Centroid
        - Spectral Bandwidth
        - Spectral Rolloff

    Returns:
        DataFrame where each row corresponds to one frame.
    """

    # ------------------------------------------------------------------------
    # MFCC FEATURES
    # ------------------------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # ------------------------------------------------------------------------
    # OTHER ACOUSTIC FEATURES
    # ------------------------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    # ------------------------------------------------------------------------
    # BUILD FEATURE DATAFRAME
    # ------------------------------------------------------------------------

    n_frames = mfcc.shape[1]
    feature_dict: dict[str, Any] = {}

    for i in range(N_MFCC):
        feature_dict[f"mfcc_{i+1}"] = mfcc[i]

    for i in range(N_MFCC):
        feature_dict[f"delta_mfcc_{i+1}"] = delta_mfcc[i]

    for i in range(N_MFCC):
        feature_dict[f"delta2_mfcc_{i+1}"] = delta2_mfcc[i]

    feature_dict["rms"] = rms[0]
    feature_dict["zcr"] = zcr[0]
    feature_dict["spectral_centroid"] = spectral_centroid[0]
    feature_dict["spectral_bandwidth"] = spectral_bandwidth[0]
    feature_dict["spectral_rolloff"] = spectral_rolloff[0]

    feature_dict["frame_index"] = np.arange(n_frames)
    feature_dict["time_seconds"] = librosa.frames_to_time(
        np.arange(n_frames),
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    return pd.DataFrame(feature_dict)


def extract_features_from_file(filepath: str) -> pd.DataFrame:
    """
    Extract all frame-level features from a single audio file.

    Returns:
        DataFrame containing metadata columns + frame-level audio features.
    """
    parsed = parse_ravdess_filename(filepath)

    if parsed is None:
        raise ValueError(f"Invalid RAVDESS filename: {filepath}")

    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    feature_df = extract_frame_features(audio, sr)

    feature_df["file_path"] = filepath
    feature_df["emotion_code"] = parsed["emotion_code"]
    feature_df["emotion_label"] = parsed["emotion_label"]
    feature_df["actor_id"] = parsed["actor_id"]

    return feature_df


# ============================================================================
# DATASET PIPELINE
# ============================================================================

def build_feature_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Build a full feature dataframe from all WAV files in data_dir.

    Args:
        data_dir: Root directory containing RAVDESS audio files.

    Returns:
        Combined pandas DataFrame (unscaled).
    """
    wav_paths = sorted(
        glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    )

    print(f"Found {len(wav_paths)} wav files")

    all_feature_frames: list[pd.DataFrame] = []

    for idx, filepath in enumerate(wav_paths, start=1):
        try:
            file_features = extract_features_from_file(filepath)
            all_feature_frames.append(file_features)
            print(f"[{idx}/{len(wav_paths)}] Processed: {Path(filepath).name}")
        except Exception as exc:
            print(f"Failed processing {filepath}: {exc}")

    return pd.concat(all_feature_frames, ignore_index=True)


# ============================================================================
# SCALING
# ============================================================================

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return acoustic feature column names (all columns except metadata)."""
    return [col for col in df.columns if col not in METADATA_COLS]


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on train acoustic features and transform both splits.

    The scaler is fit only on the training set to prevent data leakage.
    Metadata columns (file_path, emotion_label, etc.) are left untouched.

    Args:
        train_df: Training frame-level DataFrame (unscaled).
        test_df:  Test frame-level DataFrame (unscaled).

    Returns:
        Tuple of (scaled_train_df, scaled_test_df, fitted_scaler).
        The fitted scaler can be saved and reused at inference time.
    """
    feature_cols = get_feature_cols(train_df)

    scaler = StandardScaler()

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, test_scaled, scaler


# ============================================================================
# TRAIN / TEST SPLIT
# ============================================================================


def train_test_split_by_file(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Split frame-level dataframe into train/test WITHOUT leakage.
 
    Splitting is done at file (utterance) level so frames from the same
    recording never appear in both splits. Features are then StandardScaled
    (fit on train only).
 
    Args:
        df:           Full frame-level DataFrame returned by build_feature_dataframe.
        test_size:    Fraction of files to hold out for testing.
        random_state: Random seed for reproducibility.
 
    Returns:
        Tuple of (train_df, test_df, fitted_scaler).
    """
    # Unique files with their emotion label for stratification
    file_df = (
        df[["file_path", "emotion_label"]]
        .drop_duplicates(subset="file_path")
        .reset_index(drop=True)
    )
 
    train_files, test_files = train_test_split(
        file_df["file_path"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=file_df["emotion_label"].to_numpy(),
    )
 
    train_df = df[df["file_path"].isin(train_files)].reset_index(drop=True)
    test_df = df[df["file_path"].isin(test_files)].reset_index(drop=True)
 
    # Scale after splitting to avoid leakage
    train_df, test_df, scaler = scale_features(train_df, test_df)
 
    return train_df, test_df, scaler
 
 


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    df_features = build_feature_dataframe(DATA_DIR)
    print("Full shape:", df_features.shape)

    train_df, test_df, scaler = train_test_split_by_file(df_features)
    print("Train shape:", train_df.shape)
    print("Test shape: ", test_df.shape)

    train_df.to_parquet("data/processed/train_features.parquet", index=False, engine="pyarrow")
    test_df.to_parquet("data/processed/test_features.parquet", index=False, engine="pyarrow")
    print("Saved train_features.parquet and test_features.parquet")