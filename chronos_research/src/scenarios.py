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

from config import (
    DATA_DIR, CONTEXT_LENGTH, PREDICTION_LENGTH, QUANTILE_LEVELS, SEED,
    ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN, RESULTS_DIR
)
from metrics import calculate_metrics

rng = np.random.default_rng(SEED)

# ── S1 — Univariate Baseline (zero-shot, no covariates) ───────────────────────
def s1_predict(
    pipeline, 
    context_df: pd.DataFrame, 
    prediction_length: int = PREDICTION_LENGTH, 
    quantile_levels: list = QUANTILE_LEVELS,
    id_column: str = ID_COLUMN, 
    timestamp_column: str = TIMESTAMP_COLUMN, 
    target_column: str = TARGET_COLUMN,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generates a probabilistic time series forecast using the specified pipeline.
    
    Parameters:
    -----------
    pipeline : Model/Pipeline object
        The trained forecasting model pipeline (e.g., Chronos or TSFM).
    context_df : pd.DataFrame
        The historical context data used to condition the forecast.
    prediction_length : int, default PREDICTION_LENGTH
        The number of steps (hours) ahead to forecast.
    quantile_levels : list, default QUANTILE_LEVELS
        The specific quantiles to return for a probabilistic forecast.
    id_column : str, default "id"
        The column identifying distinct time series/groups.
    timestamp_column : str, default "timestamp"
        The column containing datetime information.
    target_column : str, default "target"
        The target variable column to predict.
    save_path : str, optional, default None 
        The file path (CSV) where the predictions should be saved.
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the forecast result.
    """
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length = prediction_length,
        quantile_levels = quantile_levels,
        id_column = id_column,
        timestamp_column = timestamp_column,
        target = target_column
    )

    if save_path:
        pred_df.to_csv(save_path, index=False)
        print(f"Predictions successfully created and saved to '{save_path}'.")
        
    return pred_df

def s1_evaluation(
    pred_df: pd.DataFrame, 
    ground_truth: np.ndarray, 
    id_column: str = ID_COLUMN,
    save_path: str = None
) -> pd.DataFrame:
    """
    Evaluates s1_predict forecasts by calculating RMSE and MAPE for each instance (id).
    
    Parameters:
    -----------
    pred_df : pd.DataFrame
        The forecast output from s1_predict, containing a column "0.5" 
        (median forecast) and the id_column.
    ground_truth : np.ndarray
        A 2D array of shape (num_instances, prediction_length) containing 
        the true actual values corresponding to each unique id sequentially.
    id_column : str, default "id"
        The column identifying distinct time series/groups.
    save_path : str, optional, default None  
        The file path (CSV) where the metrics DataFrame should be saved.
    Returns:
    --------
    pd.DataFrame
        A DataFrame indexed or grouped by id_column containing columns:
        ['rmse', 'mape'] representing metrics for each individual instance.
    """
    # Extract the Median Predictions
    # Group by 'id' to ensure the array shape aligns perfectly with the ground truth matrix
    forecast_median = np.array([
        group["predictions"].values for _, group in pred_df.groupby(ID_COLUMN, sort=False)
    ])

    # Calculate RMSE and MAPE for each forecast instance (each id)
    metric_records = []

    ids = list(pred_df.groupby(ID_COLUMN, sort=False).groups.keys())

    for i, ts_id in enumerate(ids):

        y_true = ground_truth[i]
        y_pred = forecast_median[i]

        rmse, mape = calculate_metrics(y_true, y_pred)

        # Dataset identifier:
        # AUS_NSW_day_0 -> AUS
        dataset = ts_id.split("_")[0]

        metric_records.append({
            "id": ts_id,
            "dataset": dataset,
            "rmse": rmse,
            "mape": mape
        })

    metrics_df = pd.DataFrame(metric_records)

    if save_path:
        metrics_df.to_csv(save_path, index=False)
        print(f"Metrics successfully created and saved to '{save_path}'.")

    return metrics_df
