"""
scenarios.py — All 4 experimental scenarios for the Panama dataset.

Uses the official Chronos-2 API from HuggingFace:
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", ...)
    pred_df  = pipeline.predict_df(context_df, future_df=future_df, ...)

Reference: https://huggingface.co/amazon/chronos-2
Paper    : arXiv:2510.15821
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from config import (
    TARGET_COL, TIMESTAMP_COL, SERIES_ID,
    PREDICTION_LENGTH, QUANTILE_LEVELS, NUM_SAMPLES,
    N_ENSEMBLE, WEATHER_NOISE_FRAC, SEED,
)

rng = np.random.default_rng(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context_df(train_df: pd.DataFrame,
                      extra_cols: list[str] = None,
                      series_id: str = SERIES_ID) -> pd.DataFrame:
    """
    Build a context DataFrame in the format Chronos2Pipeline.predict_df() expects:
        id | timestamp | target | [optional covariates ...]
    """
    cols = [TIMESTAMP_COL, TARGET_COL] + (extra_cols or [])
    ctx = train_df[cols].copy()
    ctx.columns = ["timestamp", "target"] + (extra_cols or [])
    ctx.insert(0, "id", series_id)
    return ctx


def _build_future_df(test_df: pd.DataFrame,
                     extra_cols: list[str] = None,
                     series_id: str = SERIES_ID) -> pd.DataFrame:
    """
    Build a future DataFrame for known-future covariates:
        id | timestamp | [optional covariates ...]
    (no target column — that's what we're predicting)
    """
    cols = [TIMESTAMP_COL] + (extra_cols or [])
    fut = test_df[cols].copy()
    fut.columns = ["timestamp"] + (extra_cols or [])
    fut.insert(0, "id", series_id)
    return fut


def _extract_quantiles(pred_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract q0.1, q0.5, q0.9 arrays from the predict_df() output DataFrame.
    Chronos-2 returns columns named '0.1', '0.5', '0.9' (strings).
    """
    q_lo  = pred_df["0.1"].values
    y_med = pred_df["0.5"].values
    q_hi  = pred_df["0.9"].values
    return q_lo, y_med, q_hi


# ── S1 — Univariate Baseline (zero-shot, no covariates) ───────────────────────

def run_s1(pipeline, train_df: pd.DataFrame) -> dict:
    """
    S1: Chronos-2 on raw demand only.
    Serves as the reference point for RQ1.1.
    """
    print("\n[S1] Univariate Baseline ...")

    context_df = _build_context_df(train_df)

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length = PREDICTION_LENGTH,
        quantile_levels   = QUANTILE_LEVELS,
        id_column         = "id",
        timestamp_column  = "timestamp",
        target            = "target",
    )

    q_lo, y_pred, q_hi = _extract_quantiles(pred_df)
    return {"y_pred": y_pred, "q_lo": q_lo, "q_hi": q_hi, "label": "S1_Univariate"}


# ── S2 — Covariate-Informed Forecasting ───────────────────────────────────────

def run_s2(pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame,
           covariate_cols: list[str]) -> dict:
    """
    S2: Weather (T2M, QV2M, W2M, PS) + calendar features as covariates.
    Uses Chronos-2's native known-future covariate support via future_df.

    RQ1.1, RQ1.2 — does weather integration improve over S1?
    """
    print(f"\n[S2] Covariate-Informed Forecasting  (covariates: {covariate_cols}) ...")

    context_df = _build_context_df(train_df, extra_cols=covariate_cols)
    future_df  = _build_future_df(test_df,   extra_cols=covariate_cols)

    pred_df = pipeline.predict_df(
        context_df,
        future_df         = future_df,
        prediction_length = PREDICTION_LENGTH,
        quantile_levels   = QUANTILE_LEVELS,
        id_column         = "id",
        timestamp_column  = "timestamp",
        target            = "target",
    )

    q_lo, y_pred, q_hi = _extract_quantiles(pred_df)
    return {"y_pred": y_pred, "q_lo": q_lo, "q_hi": q_hi, "label": "S2_Covariate"}


# ── S3a — Cross-City Learning ──────────────────────────────────────────────────

def run_s3a(pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame,
            city_temp_cols: list[str], covariate_cols: list[str]) -> dict:
    """
    S3a: Feed national demand + per-city temperature series together so
    Chronos-2's group-attention layers can learn shared cross-city patterns.

    Each city series is built as a separate entry (id) in context_df / future_df.
    The national demand series is the last entry; its forecast is extracted.

    RQ1.1, RQ1.2
    """
    print(f"\n[S3a] Cross-City Learning  (cities: {city_temp_cols}) ...")

    if not city_temp_cols:
        print("  [skip] No city temperature columns found in dataset.")
        return None

    # ── Build multi-series context ─────────────────────────────────────────────
    demand_min = train_df[TARGET_COL].min()
    demand_max = train_df[TARGET_COL].max()

    def scale_to_demand(s: np.ndarray) -> np.ndarray:
        """Min-max scale a city temperature to the national demand range."""
        s = s.astype(float)
        return (s - s.min()) / (s.max() - s.min() + 1e-8) * (demand_max - demand_min) + demand_min

    rows_ctx, rows_fut = [], []

    # One entry per city temperature (as a proxy demand series)
    for city_col in city_temp_cols:
        city_id = f"city_{city_col}"

        city_ctx = train_df[[TIMESTAMP_COL, city_col]].copy()
        city_ctx.columns = ["timestamp", "target"]
        city_ctx["target"] = scale_to_demand(city_ctx["target"].values)
        city_ctx.insert(0, "id", city_id)
        # Add same covariates so group attention can align
        for c in covariate_cols:
            city_ctx[c] = train_df[c].values
        rows_ctx.append(city_ctx)

        city_fut = test_df[[TIMESTAMP_COL]].copy()
        city_fut.columns = ["timestamp"]
        city_fut.insert(0, "id", city_id)
        for c in covariate_cols:
            city_fut[c] = test_df[c].values
        rows_fut.append(city_fut)

    # National demand as the target series (last — its forecast is extracted)
    rows_ctx.append(_build_context_df(train_df, extra_cols=covariate_cols))
    rows_fut.append(_build_future_df(test_df,   extra_cols=covariate_cols))

    context_df = pd.concat(rows_ctx, ignore_index=True)
    future_df  = pd.concat(rows_fut, ignore_index=True)

    pred_df = pipeline.predict_df(
        context_df,
        future_df         = future_df,
        prediction_length = PREDICTION_LENGTH,
        quantile_levels   = QUANTILE_LEVELS,
        id_column         = "id",
        timestamp_column  = "timestamp",
        target            = "target",
    )

    # Extract only the national demand forecast
    national_pred = pred_df[pred_df["id"] == SERIES_ID].reset_index(drop=True)
    q_lo, y_pred, q_hi = _extract_quantiles(national_pred)
    return {"y_pred": y_pred, "q_lo": q_lo, "q_hi": q_hi, "label": "S3a_CrossCity"}


# ── S3b — STL Decomposition Cross-Learning ────────────────────────────────────

def run_s3b(pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame,
            covariate_cols: list[str]) -> dict:
    """
    S3b: Decompose the national demand with STL (period=24 for daily pattern).
    Feed trend, seasonal, and residual as separate series alongside the original.
    Chronos-2 group-attention can learn how each component contributes.

    Two reconstructions are computed:
      - Additive: trend_fc + seasonal_fc + residual_fc
      - Direct  : forecast of the original series (within the same batch)
    The better one (lower MAPE) is returned as y_pred.

    RQ1.1, RQ1.2
    """
    print("\n[S3b] STL Decomposition Cross-Learning ...")

    target_vals = train_df[TARGET_COL].values.astype(float)

    # STL with daily period (24 h), robust to outliers
    stl_fit = STL(target_vals, period=24, robust=True).fit()
    trend    = stl_fit.trend.astype(np.float32)
    seasonal = stl_fit.seasonal.astype(np.float32)
    residual = stl_fit.resid.astype(np.float32)

    # Timestamp column shared across all component series
    timestamps = train_df[TIMESTAMP_COL].values
    fut_timestamps = test_df[TIMESTAMP_COL].values

    def _component_ctx(values: np.ndarray, series_id: str) -> pd.DataFrame:
        df = pd.DataFrame({"id": series_id, "timestamp": timestamps, "target": values})
        for c in covariate_cols:
            df[c] = train_df[c].values
        return df

    def _component_fut(series_id: str) -> pd.DataFrame:
        df = pd.DataFrame({"id": series_id, "timestamp": fut_timestamps})
        for c in covariate_cols:
            df[c] = test_df[c].values
        return df

    context_df = pd.concat([
        _component_ctx(trend,        "stl_trend"),
        _component_ctx(seasonal,     "stl_seasonal"),
        _component_ctx(residual,     "stl_residual"),
        _build_context_df(train_df,  extra_cols=covariate_cols),   # original last
    ], ignore_index=True)

    future_df = pd.concat([
        _component_fut("stl_trend"),
        _component_fut("stl_seasonal"),
        _component_fut("stl_residual"),
        _build_future_df(test_df, extra_cols=covariate_cols),
    ], ignore_index=True)

    pred_df = pipeline.predict_df(
        context_df,
        future_df         = future_df,
        prediction_length = PREDICTION_LENGTH,
        quantile_levels   = QUANTILE_LEVELS,
        id_column         = "id",
        timestamp_column  = "timestamp",
        target            = "target",
    )

    # ── Extract component forecasts ────────────────────────────────────────────
    trend_fc    = pred_df[pred_df["id"] == "stl_trend"   ]["0.5"].values
    seasonal_fc = pred_df[pred_df["id"] == "stl_seasonal"]["0.5"].values
    residual_fc = pred_df[pred_df["id"] == "stl_residual"]["0.5"].values
    national_pred = pred_df[pred_df["id"] == SERIES_ID].reset_index(drop=True)
    q_lo, y_direct, q_hi = _extract_quantiles(national_pred)

    y_additive = trend_fc + seasonal_fc + residual_fc

    return {
        "y_pred_additive": y_additive,
        "y_pred_direct"  : y_direct,
        "q_lo"           : q_lo,
        "q_hi"           : q_hi,
        "trend"          : trend,
        "seasonal"       : seasonal,
        "residual"       : residual,
        "label_additive" : "S3b_STL_Additive",
        "label_direct"   : "S3b_STL_Direct",
    }


# ── S4 — Ensemble-Based Forecasting ───────────────────────────────────────────

def run_s4(pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame,
           covariate_cols: list[str]) -> dict:
    """
    S4: Run Chronos-2 N_ENSEMBLE times, each with a slightly different
    weather realisation (Gaussian noise on T2M or similar), then average.

    This mirrors the Copernicus ERA5 EDA approach (10 realisations).
    In production: replace _perturb_weather() with real ERA5 EDA members
    downloaded via the CDS API.

    RQ1.1, RQ1.2 — does marginalising over weather uncertainty improve MAPE/RMSE?
    """
    print(f"\n[S4] Ensemble Forecasting  ({N_ENSEMBLE} realisations) ...")

    weather_col = "T2M" if "T2M" in covariate_cols else covariate_cols[0]

    def _perturb_weather(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add Gaussian noise to one weather column (one ERA5 EDA realisation)."""
        d = df.copy()
        std = d[col].std() * WEATHER_NOISE_FRAC
        d[col] = d[col] + rng.normal(0, std, len(d))
        return d

    member_forecasts = []

    for i in range(N_ENSEMBLE):
        train_m = _perturb_weather(train_df, weather_col)
        test_m  = _perturb_weather(test_df,  weather_col)

        ctx_m = _build_context_df(train_m, extra_cols=covariate_cols)
        fut_m = _build_future_df(test_m,   extra_cols=covariate_cols)

        pred_m = pipeline.predict_df(
            ctx_m,
            future_df         = fut_m,
            prediction_length = PREDICTION_LENGTH,
            quantile_levels   = QUANTILE_LEVELS,
            id_column         = "id",
            timestamp_column  = "timestamp",
            target            = "target",
        )
        member_forecasts.append(pred_m["0.5"].values)
        print(f"  member {i + 1}/{N_ENSEMBLE} done")

    ensemble_arr = np.stack(member_forecasts)            # (N, H)
    y_pred       = ensemble_arr.mean(axis=0)             # marginalised forecast
    ens_std      = ensemble_arr.std(axis=0)              # uncertainty from weather spread

    return {
        "y_pred"      : y_pred,
        "ensemble_std": ens_std,
        "label"       : "S4_Ensemble",
    }