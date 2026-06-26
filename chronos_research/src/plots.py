"""
plots.py — Visualisation helpers for forecast results and EDA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from config import RESULTS_DIR, TIMESTAMP_COLUMN, PREDICT_COLUMN

sns.set_theme(style="whitegrid", palette="muted")

# ── Boxplot ────────────────────────────────────────────────────────────────────
def plot_metrics_boxplots(merged_df):
    """
    Generates side-by-side boxplots for RMSE and MAPE comparing different datasets.
    
    Parameters:
    merged_df (pd.DataFrame): The combined DataFrame containing 'dataset', 'rmse', and 'mape' columns.
    """
    # Create side-by-side subplots using plt.subplots to configure layout without .figure()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Boxplot for RMSE
    sns.boxplot(
        data=merged_df, 
        x='dataset', 
        y='rmse', 
        ax=axes[0], 
        hue='dataset', 
        palette='Set2', 
        legend=False
    )
    axes[0].set_title('RMSE Distribution by Dataset')
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('RMSE')
    
    # 2. Boxplot for MAPE
    sns.boxplot(
        data=merged_df, 
        x='dataset', 
        y='mape', 
        ax=axes[1], 
        hue='dataset', 
        palette='Set2', 
        legend=False
    )
    axes[1].set_title('MAPE Distribution by Dataset')
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('MAPE (%)')
    
    # Automatically adjust spacing to prevent label truncations or overlaps
    plt.tight_layout()
    plt.show()
    
    # Save the visualization to a file
    plt.savefig(RESULTS_DIR / "metrics_boxplot.png")
    print(f"Box plots successfully created and saved.")
    plt.close()

# ── Distribution ───────────────────────────────────────────────────────────────
def plot_distribution(df, columns):
    """
    Plots the distribution of specified columns using histograms
    to help identify the distributional shape (Normal, Chi-sq, etc.).
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    columns (str or list): Column name(s) to plot.
    """
    # Force columns to be a list if a single string is passed
    if isinstance(columns, str):
        columns = [columns]
        
    for col in columns:
        # Create a figure 
        plt.figure(figsize=(8, 5))
        
        # 1. Histogram + KDE (Kernel Density Estimate)
        sns.histplot(df[col], kde=True, stat="density", color="royalblue", alpha=0.6)
        plt.title(f'Histogram & Density of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        
        plt.tight_layout()
        plt.show()

# ── Time Series ────────────────────────────────────────────────────────────────
def plot_time_series(data1_df, data2_matrix, save_path):
    """
    Plots the 'predictions' column from data1 and the values from data2 against the timestamps.
    
    Parameters:
    - data1_df (pd.DataFrame): DataFrame containing 'timestamp' and 'predictions' columns.
    - data2_matrix (list of lists or np.ndarray): 2D array of shape (N_DAY, 24).
    - save_path (str): File name to save the generated plot.
    """
    # 1. Convert the 2D matrix (N_DAY days x 24 hours) into a flat 1D array
    flat_data2 = np.array(data2_matrix).flatten()
    
    # 2. Ensure timestamps are in datetime format for clean plotting
    timestamps = pd.to_datetime(data1_df[TIMESTAMP_COLUMN])
    predictions = data1_df[PREDICT_COLUMN]
    
    # 3. Create the plot using subplots
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot both series
    ax.plot(timestamps, predictions, label='Predictions', color='#1f77b4', linewidth=1.5)
    ax.plot(timestamps, flat_data2, label='True Values', color='#ff7f0e', linewidth=1.5, alpha=0.8)
    
    # Formatting labels and title
    ax.set_title('Electric Load Time Series Comparison: Predictions vs True Values', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('[MW]', fontsize=12)
    
    # Grid and legend
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='upper right')
    
    # Ensure dates on the x-axis are readable and non-overlapping
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot successfully created and saved to '{save_path}'.")