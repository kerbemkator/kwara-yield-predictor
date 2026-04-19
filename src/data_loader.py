"""
data_loader.py
Handles loading, validation, and preprocessing of Kwara State crop yield data.
Designed to work with both synthetic and real FMARD/FAO datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path


FEATURE_COLS = ['rainfall_mm', 'temp_celsius', 'soil_ph', 'fertilizer_kg_ha']
TARGET_COL = 'yield_kg_ha'


def load_raw(filepath: str) -> pd.DataFrame:
    """Load CSV and perform basic validation."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(path)

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate data.
    - Drop rows with NaN in key columns
    - Remove physically impossible values
    - Log transform yield for normality
    """
    df = df.copy()
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)

    # Physical bounds
    df = df[df['rainfall_mm'].between(300, 3000)]
    df = df[df['temp_celsius'].between(15, 45)]
    df = df[df['soil_ph'].between(3.0, 9.5)]
    df = df[df['yield_kg_ha'] > 0]

    df.reset_index(drop=True, inplace=True)
    return df


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalization.
    Returns: (X_normalized, mean, std)
    Store mean/std to inverse-transform predictions.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # Prevent division by zero
    return (X - mean) / std, mean, std


def get_crop_split(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Filter dataset to a single crop type."""
    result = df[df['crop'] == crop].copy()
    if result.empty:
        available = df['crop'].unique().tolist()
        raise ValueError(f"Crop '{crop}' not found. Available: {available}")
    return result


def train_test_split_temporal(
    df: pd.DataFrame,
    test_year_cutoff: int = 2021
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split — more realistic than random split for time-series data.
    Train: years < cutoff | Test: years >= cutoff
    """
    train = df[df['year'] < test_year_cutoff].copy()
    test = df[df['year'] >= test_year_cutoff].copy()
    return train, test


def prepare_arrays(
    df: pd.DataFrame,
    normalize_features: bool = True
) -> dict:
    """
    Full pipeline: DataFrame → model-ready NumPy arrays.

    Returns a dict with:
      X, y            : raw arrays
      X_norm, mean, std : normalized arrays and params
    """
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    result = {'X': X, 'y': y, 'feature_names': FEATURE_COLS}

    if normalize_features:
        X_norm, mean, std = normalize(X)
        result.update({'X_norm': X_norm, 'norm_mean': mean, 'norm_std': std})

    return result
