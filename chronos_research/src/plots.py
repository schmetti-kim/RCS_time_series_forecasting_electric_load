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

# ── Correlation Plot ────────────────────────────────────────────────────────────────────
def plot_correlation(correlation_df, save_path : str, cmap="coolwarm", figsize=(8, 6)):
    """
    Plots a heatmap of the weather correlation results.
    
    Parameters:
    -----------
    correlation_df : pandas.DataFrame
        The DataFrame output from calculate_weather_correlations.
        Must contain a 'variable' column.
    cmap : str, default='coolwarm'
        The colormap to use for the heatmap (e.g., 'RdBu', 'viridis').
    figsize : tuple, default=(8, 6)
        The width and height of the figure in inches.
    """
    # 1. Set the 'variable' column as the row index
    corr_plot = correlation_df.set_index("variable")
    
    # 2. Automatically detect the correlation columns (city pairs)
    columns_to_plot = corr_plot.columns.tolist()
    num_cols = len(columns_to_plot)
    
    # 3. Create the plot
    plt.figure(figsize=figsize)
    
    img = plt.imshow(
        corr_plot[columns_to_plot],
        cmap=cmap,
        vmin=-1,
        vmax=1
    )
    
    # 4. Set dynamic tick labels based on the data
    plt.xticks(
        range(num_cols),
        columns_to_plot,
        rotation=45
    )
    plt.yticks(
        range(len(corr_plot)),
        corr_plot.index
    )
    
    # 5. Add colorbar and formatting
    plt.colorbar(img, label="Pearson correlation")
    plt.tight_layout()
    
    # 6. Save the figure (Do this BEFORE plt.show())
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully created and saved to '{save_path}'.")
    
    # 7. Show and close
    plt.show()
    plt.close()

# ── Variable Trajectory Comparison ─────────────────────
def plot_comparison(df1, df2, column_name, label1='Dataset 1', label2='Dataset 2'):
    """
    Plots a time series comparison of a specific column from two dataframes.
    
    Parameters:
    df1 (pd.DataFrame): First dataframe
    df2 (pd.DataFrame): Second dataframe
    column_name (str): The name of the column to plot (e.g., 'temperature_2m')
    label1, label2 (str, optional): Custom labels for the legend
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # 1. Extract and convert x-axis data to datetime
    if TIMESTAMP_COLUMN:
        x1 = pd.to_datetime(df1[TIMESTAMP_COLUMN])
        x2 = pd.to_datetime(df2[TIMESTAMP_COLUMN])
    else:
        # Defaults to using the DataFrame index if no date column is provided
        x1 = pd.to_datetime(df1.index)
        x2 = pd.to_datetime(df2.index)

    # 2. Plot the data
    ax.plot(x1, df1[column_name], label=label1, color='#1f77b4', linewidth=1.5)
    ax.plot(x2, df2[column_name], label=label2, color='#ff7f0e', linewidth=1.5)

    # 3. Format the x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate() # Tilts dates to prevent overlapping

    # 4. Labels and styling
    ax.set_title(f'{column_name} Comparison', fontsize=14, pad=12)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel(column_name, fontsize=11)
    
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 5. Display plot
    plt.show()

# ── Covariate Selection Progression Plot ──────────────────────────────────────
def plot_covariate_selection_progression(
    result_dfs: list,
    stage_names: list,
    save_path: str = None,
    metric: str = "mean_mape",
    figsize=(12, 5)
):
    """
    Plots mean MAPE progression across the hierarchical covariate selection process.

    Each stage is displayed sequentially with vertical separators:
    Group 1 -> Group 2 -> Group 3 -> Group 4 -> Backward Elimination.
    """
    # Combine all results
    plot_df = pd.concat(result_dfs, ignore_index=True)

    # Create sequential model index for plotting
    plot_df["plot_index"] = range(len(plot_df))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        plot_df["plot_index"],
        plot_df[metric],
        marker="o",
        markersize=4
    )

    # Stage boundaries
    stage_lengths = [len(df) for df in result_dfs]

    cumulative = 0
    for i, length in enumerate(stage_lengths[:-1]):
        cumulative += length
        ax.axvline(
            cumulative - 0.5,
            linestyle="--",
            linewidth=1
        )

    # Stage labels
    stage_centers = []
    cumulative = 0

    for length in stage_lengths:
        stage_centers.append(cumulative + (length - 1) / 2)
        cumulative += length

    for center, name in zip(stage_centers, stage_names):
        ax.text(
            center,
            ax.get_ylim()[1],
            name,
            ha="center",
            va="bottom"
        )

    # Highlight selected configurations if available
    selected_models = []

    for df in result_dfs:
        if "mean_mape" in df.columns:
            selected_models.append(df["mean_mape"].idxmin())

    ax.set_xlabel("Sequential model evaluation index")
    ax.set_ylabel("Mean MAPE")
    ax.set_title("Hierarchical Covariate Selection Progression")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300
        )
        print(f"Plot saved to '{save_path}'.")

    plt.show()