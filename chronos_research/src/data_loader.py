"""
data_loader.py — Load and preprocess the Panama electricity dataset.

Download dataset once with:
    kaggle datasets download -d saurabhshahane/electricity-load-forecasting \
        -p data/ --unzip
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_DIR, TIMESTAMP_COL, TARGET_COL,
    WEATHER_COLS, CITY_TEMP_COLS, PREDICTION_LENGTH,
)


# ── Calendar features ──────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar encodings (no data leakage — pure timestamp math)."""
    d = df.copy()
    dt = d[TIMESTAMP_COL]
    d["hour_sin"]   = np.sin(2 * np.pi * dt.dt.hour       / 24)
    d["hour_cos"]   = np.cos(2 * np.pi * dt.dt.hour       / 24)
    d["dow_sin"]    = np.sin(2 * np.pi * dt.dt.dayofweek  / 7)
    d["dow_cos"]    = np.cos(2 * np.pi * dt.dt.dayofweek  / 7)
    d["month_sin"]  = np.sin(2 * np.pi * dt.dt.month      / 12)
    d["month_cos"]  = np.cos(2 * np.pi * dt.dt.month      / 12)
    d["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    return d

CALENDAR_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend",
]


# ── Main loader ────────────────────────────────────────────────────────────────

def load_panama(path: Path = DATA_DIR / "raw" / "continuous_dataset.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    train_df, test_df
        Both DataFrames have calendar + weather columns added.
        test_df is the last PREDICTION_LENGTH rows.
    """
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Keep only columns that actually exist in this version of the dataset
    present_weather = [c for c in WEATHER_COLS   if c in df.columns]
    present_city    = [c for c in CITY_TEMP_COLS  if c in df.columns]

    # Drop rows with missing target; forward-fill covariate NaNs
    df = df.dropna(subset=[TARGET_COL])
    df = df.ffill()

    df = add_calendar_features(df)

    test_df  = df.tail(PREDICTION_LENGTH).copy().reset_index(drop=True)
    train_df = df.iloc[:-PREDICTION_LENGTH].copy().reset_index(drop=True)

    print(f"[data]  rows — train: {len(train_df):,}  |  test: {len(test_df)}")
    print(f"[data]  weather cols : {present_weather}")
    print(f"[data]  city cols    : {present_city}")
    print(f"[data]  test window  : {test_df[TIMESTAMP_COL].iloc[0]}  →  {test_df[TIMESTAMP_COL].iloc[-1]}")

    return train_df, test_df


def get_covariate_cols(train_df: pd.DataFrame) -> list[str]:
    """Return weather + calendar columns that are actually in the DataFrame."""
    present_weather = [c for c in WEATHER_COLS if c in train_df.columns]
    return present_weather + CALENDAR_COLS


def get_city_cols(train_df: pd.DataFrame) -> list[str]:
    return [c for c in CITY_TEMP_COLS if c in train_df.columns]