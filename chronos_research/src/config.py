"""
config.py — Central config for the Panama Chronos-2 pipeline.
Change values here; everything else reads from this file.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent  # Goes from src/ up to panama/
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────
# DATA_FILE   = DATA_DIR / "continuous_dataset.csv"   # Kaggle Panama dataset
TIMESTAMP_COL = "datetime"
TARGET_COL    = "nat_demand"                        # national demand (MW)
SERIES_ID     = "panama_national"

# Weather columns present in the Panama dataset (NASA MERRA-2)
WEATHER_COLS  = ["T2M", "QV2M", "TQL", "W2M", "PS"]
CITY_TEMP_COLS = ["T2M_toc", "T2M_san", "T2M_dav"]  # per-city temperatures

# ── Forecasting ────────────────────────────────────────────────────────────────
PREDICTION_LENGTH = 24              # 24 h = 1 day ahead  (change to 168 for 1 week)
QUANTILE_LEVELS   = [0.1, 0.5, 0.9]
NUM_SAMPLES       = 20              # stochastic draws per forecast call

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "amazon/chronos-2"      # official HuggingFace model ID

# ── Ensemble (S4) ──────────────────────────────────────────────────────────────
N_ENSEMBLE          = 10            # mirrors 10 ERA5 EDA realisations
WEATHER_NOISE_FRAC  = 0.02          # std-fraction of noise per realisation
                                    # replace generate_realisation() with real ERA5 members

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42