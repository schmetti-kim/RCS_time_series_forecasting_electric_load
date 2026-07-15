"""
config.py — Central config for Chronos-2 pipeline.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent  # Goes from src/ up to panama/
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
ID_COLUMN = "id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target"
PREDICT_COLUMN = "predictions"
WEATHER_VARS = [
            "temperature_2m", "dew_point_2m", "precipitation", "pressure_msl", 
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", 
            "wind_speed_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", 
            "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"
        ]

WEATHER_VARS_ENSEMBLE = [
            "temperature_2m_spread", "dew_point_2m_spread", "precipitation_spread", "pressure_msl_spread", 
            "cloud_cover_low_spread", "cloud_cover_mid_spread", "cloud_cover_high_spread", "wind_speed_10m_spread", 
            "wind_speed_100m_spread", "wind_gusts_10m_spread", "soil_temperature_0_to_7cm_spread", "soil_temperature_7_to_28cm_spread", 
            "soil_moisture_0_to_7cm_spread", "soil_moisture_7_to_28cm_spread"
]

# ── Forecasting ────────────────────────────────────────────────────────────────
CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 24              # 24 h = 1 day ahead  (change to 168 for 1 week)
N_DAYS = 1095 # temporarily set to 30 due to time constraints; will be set 365 x 3 eventually
QUANTILE_LEVELS   = [0.1, 0.5, 0.9]

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "amazon/chronos-2"      # official HuggingFace model ID

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42