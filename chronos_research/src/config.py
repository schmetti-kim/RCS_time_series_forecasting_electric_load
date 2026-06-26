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

# ── Forecasting ────────────────────────────────────────────────────────────────
CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 24              # 24 h = 1 day ahead  (change to 168 for 1 week)
N_DAYS = 1095 # temporarily set to 30 due to time constraints; will be set 365 x 3 eventually
QUANTILE_LEVELS   = [0.1, 0.5, 0.9]

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "amazon/chronos-2"      # official HuggingFace model ID

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42