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
from itertools import combinations

from config import (
    DATA_DIR, CONTEXT_LENGTH, PREDICTION_LENGTH, QUANTILE_LEVELS, SEED,
    ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN, RESULTS_DIR, N_DAYS, GROUP1_VARS, 
    ENCODING_CANDIDATES
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

# ── S2 — Covariate informed forecasting ────────────────────────────────────────
def s2_predict(
    pipeline, 
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    prediction_length: int = PREDICTION_LENGTH, 
    quantile_levels: list = QUANTILE_LEVELS,
    id_column: str = ID_COLUMN, 
    timestamp_column: str = TIMESTAMP_COLUMN, 
    target_column: str = TARGET_COLUMN,
    save_path: str = None
) -> pd.DataFrame:
    
    pred_df = pipeline.predict_df(
        context_df, 
        future_df = future_df,
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

# ── S3 — cross-region forecasting ────────────────────────────────────────
def s3_predict(
    pipeline, 
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    prediction_length: int = PREDICTION_LENGTH, 
    quantile_levels: list = QUANTILE_LEVELS,
    id_column: str = ID_COLUMN, 
    timestamp_column: str = TIMESTAMP_COLUMN, 
    target_column: str = TARGET_COLUMN,
    save_path: str = None
) -> pd.DataFrame:
    
    # Initialize empty DataFrame to accumulate results
    pred_df = pd.DataFrame()

    for day in range(N_DAYS):
        # Filter for the specific day across all states
        # This selects any ID ending with _day_0, _day_1, etc.
        context_subset = context_df[context_df[ID_COLUMN].str.endswith(f"_day_{day}")]
        future_subset = future_df[future_df[ID_COLUMN].str.endswith(f"_day_{day}")]
        
        pred_subset = pipeline.predict_df(
            context_subset, 
            future_df = future_subset,
            prediction_length = prediction_length,
            quantile_levels = quantile_levels,
            id_column = id_column,
            timestamp_column = timestamp_column,
            target = target_column,
            cross_learning = True
        )

        # Append pred_subset to pred_df
        pred_df = pd.concat([pred_df, pred_subset], ignore_index=True)
    
    if save_path:
        pred_df.to_csv(save_path, index=False)
        print(f"Predictions successfully created and saved to '{save_path}'.")
        
    return pred_df

# ── covariates selection process ──────────────────────────────────────────
def hierarchical_exhaustive_search(
    pipeline, 
    context_df: pd.DataFrame, 
    future_df: pd.DataFrame, 
    ground_truth: np.ndarray,
    group_covariates: list,
    base_covariates: list = None,
    stage_name: str = "Stage 1",
    start_model_idx: int = 0,
    prediction_length: int = PREDICTION_LENGTH,
    id_column: str = ID_COLUMN, 
    timestamp_column: str = TIMESTAMP_COLUMN, 
    target_column: str = TARGET_COLUMN,
    practical_mape_tolerance = 1e-2,
    save_path: str = None
) -> tuple[pd.DataFrame, list]:
    """
    Evaluates all 2^N subsets. Models whose mean MAPE lies within the specified 
    tolerance of the best-performing model are considered practically equivalent, 
    and the simplest model (fewest covariates) is selected.
    """
    base_covariates = base_covariates or []
    results = []
    
    base_context_cols = [timestamp_column, target_column, id_column]
    base_future_cols = [timestamp_column, id_column]
    
    print(f"\n--- Running Exhaustive Search for {stage_name} ---")
    
    # 1. Generate all possible subsets (size 0 to N)
    all_subsets = []
    for r in range(len(group_covariates) + 1):
        for subset in combinations(group_covariates, r):
            all_subsets.append(list(subset))
            
    current_model_idx = start_model_idx
    
    # 2. Iterate through every combination
    for subset in all_subsets:
        model_name = f"M{current_model_idx}"
        current_eval_covariates = base_covariates + subset
        
        print(f"Evaluating {model_name} (Covariates: {current_eval_covariates if current_eval_covariates else 'None'})")
        
        context_cols = base_context_cols + current_eval_covariates
        future_cols = base_future_cols + current_eval_covariates
        
        pred_df = s2_predict(
            pipeline=pipeline, 
            context_df=context_df[context_cols], 
            future_df=future_df[future_cols],
            prediction_length = prediction_length,
            quantile_levels = [0.5],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=target_column
        )
        
        metrics_df = s1_evaluation(pred_df, ground_truth)
        
        results.append({
            "stage": stage_name,
            "model": model_name,
            "covariates_tested": tuple(subset),
            "cumulative_covariates": tuple(current_eval_covariates),
            "mean_mape": metrics_df["mape"].mean(),
            "mean_rmse": metrics_df["rmse"].mean(),
            "num_covariates": len(current_eval_covariates)
        })
        
        current_model_idx += 1
        
    # 3. Compile Results and Select the Best Configuration (G*)
    results_df = pd.DataFrame(results)
    
    # Best observed MAPE
    best_mape = results_df["mean_mape"].min()

    # Keep all models whose MAPE is within the tolerance
    best_candidates = results_df[
        results_df["mean_mape"] <= best_mape + practical_mape_tolerance
    ]

    # Among statistically/practically equivalent models,
    # prefer the simplest (fewest covariates).
    best_row = (
        best_candidates
        .sort_values(
            by=["num_covariates", "mean_mape"],
            ascending=[True, True]
        )
        .iloc[0]
    )

    best_covariates = list(best_row["cumulative_covariates"])
    best_model = best_row["model"]
    
    print(f"\nExhaustive search complete for {stage_name}.")
    print(f"Selected Configuration ({stage_name}*): {best_model} -> {best_covariates}")
    print(f"Best Mean MAPE: {best_row['mean_mape']:.4f}\n")
    
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Results successfully saved to '{save_path}'.")
        
    return results_df, best_row

def evaluate_diverse_encodings(
    pipeline,
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    ground_truth: np.ndarray,
    base_covariates: list,
    encoding_candidates: dict,
    stage_name: str = "Group 2",
    start_model_idx: int = 0,
    prediction_length: int = PREDICTION_LENGTH,
    id_column: str = ID_COLUMN,
    timestamp_column: str = TIMESTAMP_COLUMN,
    target_column: str = TARGET_COLUMN,
    practical_mape_tolerance: float = 1e-2,
    save_path: str = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Evaluates alternative day-of-week encodings on top of the selected Group 1
    configuration. Models whose mean MAPE lies within the specified tolerance
    of the best-performing model are considered practically equivalent, and
    the simplest encoding is selected.
    """

    results = []

    base_context_cols = [timestamp_column, target_column, id_column]
    base_future_cols = [timestamp_column, id_column]

    print(f"\n--- Running Calendar Encoding Selection for {stage_name} ---")

    current_model_idx = start_model_idx

    for encoding_name, encoding_covariates in encoding_candidates.items():

        model_name = f"M{current_model_idx}"
        current_eval_covariates = base_covariates + encoding_covariates

        print(
            f"Evaluating {model_name} "
            f"({encoding_name}: "
            f"{encoding_covariates if encoding_covariates else 'None'})"
        )

        context_cols = base_context_cols + current_eval_covariates
        future_cols = base_future_cols + current_eval_covariates

        pred_df = s2_predict(
            pipeline=pipeline,
            context_df=context_df[context_cols],
            future_df=future_df[future_cols],
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )

        metrics_df = s1_evaluation(pred_df, ground_truth)

        results.append({
            "stage": stage_name,
            "model": model_name,
            "encoding": encoding_name,
            "encoding_covariates": tuple(encoding_covariates),
            "cumulative_covariates": tuple(current_eval_covariates),
            "mean_mape": metrics_df["mape"].mean(),
            "mean_rmse": metrics_df["rmse"].mean(),
            "num_covariates": len(current_eval_covariates),
        })

        current_model_idx += 1

    results_df = pd.DataFrame(results)

    best_mape = results_df["mean_mape"].min()

    best_candidates = results_df[
        results_df["mean_mape"] <= best_mape + practical_mape_tolerance
    ]

    best_row = (
        best_candidates
        .sort_values(
            by=["num_covariates", "mean_mape"],
            ascending=[True, True],
        )
        .iloc[0]
    )

    print(f"\nCalendar encoding selection complete for {stage_name}.")
    print(
        f"Selected Encoding ({stage_name}*): "
        f"{best_row['model']} -> {best_row['encoding']}"
    )
    print(f"Best Mean MAPE: {best_row['mean_mape']:.4f}\n")

    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Results successfully saved to '{save_path}'.")

    return results_df, best_row

def backward_one_out_elimination(
    pipeline,
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    ground_truth: np.ndarray,
    selected_covariates: list,
    stage_name: str = "Backward Elimination",
    start_model_idx: int = 0,
    prediction_length: int = PREDICTION_LENGTH,
    id_column: str = ID_COLUMN,
    timestamp_column: str = TIMESTAMP_COLUMN,
    target_column: str = TARGET_COLUMN,
    practical_mape_tolerance: float = 1e-2,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Performs one-out elimination analysis on the selected covariates.
    Each covariate is removed individually and evaluated. The function
    identifies covariates whose individual removal does not increase mean
    MAPE beyond the specified practical tolerance.
    """

    selected_covariates = list(selected_covariates)
    results = []

    base_context_cols = [timestamp_column, target_column, id_column]
    base_future_cols = [timestamp_column, id_column]

    print(f"\n--- Running {stage_name} ---")

    # Evaluate the full model
    full_context_cols = base_context_cols + selected_covariates
    full_future_cols = base_future_cols + selected_covariates

    pred_df = s2_predict(
        pipeline=pipeline,
        context_df=context_df[full_context_cols],
        future_df=future_df[full_future_cols],
        prediction_length=prediction_length,
        quantile_levels=[0.5],
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )

    metrics_df = s1_evaluation(pred_df, ground_truth)
    baseline_mape = metrics_df["mape"].mean()

    current_model_idx = start_model_idx

    # Remove one covariate at a time
    for covariate in selected_covariates:

        remaining_covariates = [
            c for c in selected_covariates if c != covariate
        ]

        model_name = f"M{current_model_idx}"

        print(f"Evaluating {model_name} (Remove: {covariate})")

        context_cols = base_context_cols + remaining_covariates
        future_cols = base_future_cols + remaining_covariates

        pred_df = s2_predict(
            pipeline=pipeline,
            context_df=context_df[context_cols],
            future_df=future_df[future_cols],
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )

        metrics_df = s1_evaluation(pred_df, ground_truth)

        mean_mape = metrics_df["mape"].mean()

        removable = mean_mape <= baseline_mape + practical_mape_tolerance

        results.append({
            "stage": stage_name,
            "model": model_name,
            "removed_covariate": covariate,
            "remaining_covariates": tuple(remaining_covariates),
            "mean_mape": mean_mape,
            "mean_rmse": metrics_df["rmse"].mean(),
            "removable": removable,
        })

        current_model_idx += 1

    results_df = pd.DataFrame(results)

    print(f"\n{stage_name} complete.")
    removable_covariates = results_df.loc[
        results_df["removable"],
        "removed_covariate"
    ].tolist()

    if removable_covariates:
        print(f"Covariates removable within the specified tolerance: {removable_covariates}")
    else:
        print("No covariates are removable within the specified tolerance.")

    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Results successfully saved to '{save_path}'.")

    return results_df