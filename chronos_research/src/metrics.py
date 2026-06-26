"""
metrics.py — Evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return rmse, mape

# def save_results(results: list[dict], results_dir: Path, dataset: str = "Panama"):
#     """Save CSV + JSON (JSON used by teammates for RQ2 cross-dataset merge)."""
#     df = pd.DataFrame(results)

#     csv_path  = results_dir / f"{dataset.lower()}_results.csv"
#     json_path = results_dir / f"{dataset.lower()}_results_rq2.json"

#     df.to_csv(csv_path, index=False)

#     # JSON export with dataset label — team merges all 3 datasets here
#     df.insert(0, "dataset", dataset)
#     df.to_json(json_path, orient="records", indent=2)

#     print(f"\n[saved]  {csv_path}")
#     print(f"[saved]  {json_path}  ← share with Kim & Saxena for RQ2")
#     return df

def compute_summary_statistics(df, columns):
    """
    Computes summary statistics for evaluation metrics.
    
    Parameters:
    df (pd.DataFrame): The evaluation dataframe containing metrics.
    columns (list): List of columns to compute statistics for.
    
    Returns:
    pd.DataFrame: A formatted dataframe containing summary statistics.
    """
    # Force columns to be a list if a single string is passed
    if isinstance(columns, str):
        columns = [columns]
        
    # Extract and ensure we force a DataFrame structure, even for a single column
    # (Using df[columns] where columns is a list naturally returns a DataFrame)
    sub_df = df[columns]

    # Compute basic aggregations
    stats = sub_df.agg(['count', 'mean', 'median', 'var', 'min', 'max'])
    
    # Calculate quantiles separately to avoid index naming conflicts across pandas versions
    q1 = sub_df.quantile(0.25)
    q3 = sub_df.quantile(0.75)
    
    # Append quantiles to the summary dataframe
    stats.loc['q1'] = q1
    stats.loc['q3'] = q3
    
    # Reorder index labels for logical readability
    ordered_indices = ['count', 'mean', 'median', 'var', 'min', 'q1', 'q3', 'max']
    stats = stats.reindex(ordered_indices)
        
    return stats