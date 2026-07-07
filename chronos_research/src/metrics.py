"""
metrics.py — Evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path
from config import WEATHER_VARS

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return rmse, mape

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

def calculate_weather_correlations(df, suffixes, labels, weather_vars = WEATHER_VARS):
    """
    Calculates the correlation between three locations for a given list of weather variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The merged weather DataFrame containing the suffixed columns.
    weather_vars : list of str
        The base names of the weather variables to loop through.
    suffixes : tuple or list of str, e.g.,('_rig', '_dgp', '_lpx')
        The suffixes used in the DataFrame columns for each location.
    labels : tuple or list of str, e.g.,('Riga', 'Daugavpils', 'Liepaja')
        The human-readable names for the locations used to label the output columns.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the correlation results for each variable across the 3 location pairs.
    """
    if len(suffixes) != 3 or len(labels) != 3:
        raise ValueError("Both 'suffixes' and 'labels' must contain exactly 3 elements.")
        
    sfx1, sfx2, sfx3 = suffixes
    lbl1, lbl2, lbl3 = labels
    
    correlation_results = []

    for var in weather_vars:
        correlation_results.append({
            "variable": var,
            f"{lbl1}-{lbl2}": df[f"{var}{sfx1}"].corr(df[f"{var}{sfx2}"]),
            f"{lbl1}-{lbl3}": df[f"{var}{sfx1}"].corr(df[f"{var}{sfx3}"]),
            f"{lbl2}-{lbl3}": df[f"{var}{sfx2}"].corr(df[f"{var}{sfx3}"])
        })

    return pd.DataFrame(correlation_results)