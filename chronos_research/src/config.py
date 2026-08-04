"""
config.py — Central config for Chronos-2 pipeline.
"""

from pathlib import Path
from itertools import combinations

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
WEATHER_VARS_ENSEMBLE = []
WEATHER_VARS_FORECAST = []
DROP_VARS = [
  "dew_point_2m",
  "pressure_msl",
  "wind_speed_10m",
  "wind_speed_100m",
  "wind_gusts_10m",
  "precipitation",
  "soil_temperature_0_to_7cm",
  "soil_temperature_7_to_28cm",
  "soil_moisture_0_to_7cm",
  "soil_moisture_7_to_28cm",
  "cloud_cover_low",
  "cloud_cover_mid",
  "cloud_cover_high",
  "dow", "dow_0", "dow_1", "dow_2", 
  "dow_3", "dow_4", "dow_5", "dow_6", 
  "dow_sin", "dow_cos"]

SELECTED_WEATHER_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "pressure_msl",
    "wind_speed_10m",      # representative wind variable
    "cloud_cover_low",     # representative cloud variable
]

# GROUP1_CANDIDATES = {}

# for holiday in [[], ["holiday"]]:
#     for weekend in [[], ["weekend"]]:
#         for dow_name, dow_cols in {
#             "none": [],
#             "integer": ["dow"],
#             "one_hot": [
#                 "dow_0", "dow_1", "dow_2",
#                 "dow_3", "dow_4", "dow_5", "dow_6"
#             ],
#             "cyclic": ["dow_sin", "dow_cos"]
#         }.items():
#             name = f"h{len(holiday)}_w{len(weekend)}_{dow_name}"
#             GROUP1_CANDIDATES[name] = holiday + weekend + dow_cols

GROUP1_CANDIDATES = {}

candidate_idx = 0

for r in range(len(SELECTED_WEATHER_VARS) + 1):
    for weather_subset in combinations(SELECTED_WEATHER_VARS, r):

        for holiday in [[], ["holiday"]]:
            for weekend in [[], ["weekend"]]:
                for dow_name, dow_cols in {
                    "none": [],
                    "integer": ["dow"],
                    "one_hot": [
                        "dow_0", "dow_1", "dow_2",
                        "dow_3", "dow_4", "dow_5", "dow_6"
                    ],
                    "cyclic": ["dow_sin", "dow_cos"]
                }.items():

                    name = (
                        f"h{len(holiday)}_"
                        f"w{len(weekend)}_"
                        f"{dow_name}_"
                        f"ws_{'_'.join(weather_subset) if weather_subset else 'none'}"
                    )

                    GROUP1_CANDIDATES[name] = (
                        holiday
                        + weekend
                        + dow_cols
                        + list(weather_subset)
                    )

GROUP2_CANDIDATES = {
    name: covariates
    for name, covariates in GROUP1_CANDIDATES.items()
    if any(var in covariates for var in SELECTED_WEATHER_VARS)
}

# ── Forecasting ────────────────────────────────────────────────────────────────
CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 24              # 24 h = 1 day ahead  (change to 168 for 1 week)
N_DAYS = 1095 # temporarily set to 30 due to time constraints; will be set 365 x 3 eventually
QUANTILE_LEVELS   = [0.1, 0.3, 0.7, 0.9]

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "amazon/chronos-2"      # official HuggingFace model ID

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42