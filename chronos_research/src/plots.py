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
from adjustText import adjust_text
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
    axes[0].set_ylabel('RMSE (MW)')
    
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

    # ── Model 0 Baseline Horizontal Line ───────────────────────────────
    model_0_val = plot_df.loc[0, metric]
    ax.axhline(y=model_0_val, color="gray", linestyle="--", alpha=0.7, label="Univariate Baseline")

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

    # ── Highlight cumulative best configuration & horizontal line ─────
    cumulative = 0
    running_best_val = float("inf")
    running_best_global_idx = None
    running_best_config = None

    for stage_name, df in zip(stage_names, result_dfs):
        if metric in df.columns:
            # 1. Find local best in the current stage
            local_min_idx = df[metric].idxmin()
            local_min_val = df.loc[local_min_idx, metric]
            
            # 2. Update cumulative best if current stage beats previous best
            if local_min_val < running_best_val:
                running_best_val = local_min_val
                running_best_global_idx = cumulative + local_min_idx
                running_best_config = df.loc[local_min_idx, "configuration"] if "configuration" in df.columns else "N/A"

            print(f"[{stage_name}] Cumulative Best {metric}: {running_best_val:.4f} | Configuration: {running_best_config}")

            start_idx = cumulative
            end_idx = cumulative + len(df) - 1

            # Highlight the cumulative best point seen so far
            ax.scatter(running_best_global_idx, running_best_val, color="red", s=40, zorder=5)

            # Draw the horizontal line for the current stage at the cumulative best level
            ax.hlines(
                y=running_best_val, 
                xmin=start_idx,      # Use xmin=0 if you prefer the line to extend from the very start
                xmax=end_idx, 
                colors="red", 
                linestyles=":", 
                linewidth=1.2
            )

        cumulative += len(df)

    ax.set_xlabel("Configuration Index")
    ax.set_ylabel("Mean MAPE (%)")
    ax.set_title("Covariate Selection Progression", y=1.12)

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

# ── Metric Statistics Progression Plot ──────────────────────────────────────
def plot_metric_progression(
    merged_table: pd.DataFrame,
    metric: str = "rmse",
    stat: str = "mean",
    mode: str = "indexed",  # Options: 'indexed', 'subplots', 'raw'
    figsize: tuple = None,
):
    """Visualizes metric progression with options to highlight subtle variations.

    Parameters:
    - merged_table: DataFrame containing summary stats across experiments.
    - metric: 'rmse' or 'mape'.
    - stat: 'mean', 'median', or 'variance' / 'var'.
    - mode:
        * 'indexed'  : Shows % change relative to Exp 1 (baseline = 0%).
                       Best for comparing relative trajectory across series on one plot.
        * 'subplots' : 1x3 faceted grid with independent y-axes.
                       Best for inspecting exact absolute values per dataset.
        * 'raw'      : Standard single plot with raw values on a shared axis.
    - figsize: Optional tuple for figure dimensions.
    """
    metric = metric.lower()
    stat_key = "var" if stat.lower() in ["variance", "var"] else stat.lower()

    if stat_key not in merged_table.index:
        raise ValueError(
            f"Stat '{stat}' not in index. Options: {list(merged_table.index)}"
        )

    experiments = [
        ("Exp 1 (Uni.)", "{country}_uni_{metric}"),
        ("Exp 2 (Cov.)", "{country}_cov_{metric}"),
        ("Exp 3-1 (Cross)", "{country}_s3_{metric}_before_holiday"),
        ("Exp 3-2 (Cross-H)", "{country}_s3_{metric}"),
    ]
    x_labels = [exp[0] for exp in experiments]

    datasets = {
        "Panama": {"code": "pan", "color": "#1f77b4", "marker": "o"},
        "Australia": {"code": "aus", "color": "#ff7f0e", "marker": "s"},
        "Latvia": {"code": "lat", "color": "#2ca02c", "marker": "^"},
    }

    stat_label = "Variance" if stat_key == "var" else stat.capitalize()
    metric_label = metric.upper()

    # Extract data matrix: shape (3 datasets, 4 experiments)
    data = {}
    for name, info in datasets.items():
        cols = [
            exp[1].format(country=info["code"], metric=metric)
            for exp in experiments
        ]
        data[name] = merged_table.loc[stat_key, cols].values.astype(float)

    # ---------------------------------------------------------
    # MODE 1: Indexed (% Change from Baseline)
    # ---------------------------------------------------------
    if mode == "indexed":
        fig, ax = plt.subplots(figsize=figsize or (9, 5))
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)

        texts = []
        for name, values in data.items():
            baseline = values[0]
            pct_change = (
                (values - baseline) / baseline
            ) * 100  # % change relative to Exp 1

            ax.plot(
                x_labels,
                pct_change,
                label=name,
                linewidth=1,
                markersize=2,
                color=datasets[name]["color"],
                marker=datasets[name]["marker"],
            )

            # Direct value annotations to see subtle differences
            # Inside the datasets loop:
            for x, y, raw in zip(x_labels, pct_change, values):
                txt = ax.text(
                    x,
                    y,
                    f"{raw:.2f}\n({y:+.1f}%)",
                    fontsize=8.5,
                    color=datasets[name]["color"],
                    weight="semibold",
                    ha="center",
                    va="center",
                )
                texts.append(txt)

        adjust_text(texts, ax=ax, autoalign="y", only_move={"text": "y"})

        ax.set_title(
            f"Relative {stat_label} {metric_label} Change vs. Univariate (Exp 1 = 0%)",
            fontsize=12,
            pad=14,
        )
        ax.set_ylabel(
            f"% Change in {stat_label} {metric_label}",
            fontsize=10.5,
            labelpad=8,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Dataset", frameon=True, loc="best")

        # Give breathing room for annotations
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(
            y_min - abs(y_min) * 0.2 - 2, y_max + max(abs(y_max) * 0.2, 5)
        )

    # ---------------------------------------------------------
    # MODE 2: Independent Subplots (Faceting)
    # ---------------------------------------------------------
    elif mode == "subplots":
        fig, axes = plt.subplots(1, 3, figsize=figsize or (15, 4.5), sharex=True)

        for ax, (name, values) in zip(axes, data.items()):
            color = datasets[name]["color"]
            marker = datasets[name]["marker"]

            ax.plot(
                x_labels,
                values,
                label=name,
                linewidth=1,
                markersize=2,
                color=color,
                marker=marker,
            )

            for x, y in zip(x_labels, values):
                ax.annotate(
                    f"{y:.2f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=9,
                    weight="bold",
                )

            ax.set_title(f"{name}", fontsize=11, weight="bold")
            ax.set_ylabel(f"{stat_label} {metric_label}", fontsize=9.5)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="x", rotation=15)

            # Expand y-limits slightly to prevent text clipping
            y_span = values.max() - values.min()
            pad = (
                y_span * 0.35 if y_span > 0 else values.mean() * 0.05
            )  # prevent flat-line crash
            ax.set_ylim(values.min() - pad, values.max() + pad)

        fig.suptitle(
            f"{stat_label} {metric_label} Progression (Independent Y-Axes)",
            fontsize=13,
            y=1.03,
        )

    # ---------------------------------------------------------
    # MODE 3: Raw
    # ---------------------------------------------------------
    else:
        fig, ax = plt.subplots(figsize=figsize or (8, 5))

        texts = []
        for name, values in data.items():
            ax.plot(
                x_labels,
                values,
                label=name,
                linewidth=1,
                markersize=2,
                color=datasets[name]["color"],
                marker=datasets[name]["marker"],
            )

            # Inside the datasets loop:
            for x, y in zip(x_labels, values):
                txt = ax.text(
                    x,
                    y,
                    f"{y:.2f}",
                    fontsize=8.5,
                    color=datasets[name]["color"],
                    ha="center",
                    va="center",
                )
                texts.append(txt)

        # Right after the datasets loop finishes (before ax.set_title):
        adjust_text(texts, ax=ax, autoalign="y", only_move={"text": "y"})

        ax.set_title(
            f"{stat_label} {metric_label} Progression across Experiments",
            fontsize=12,
            pad=12,
        )
        ax.set_ylabel(f"{stat_label} {metric_label}", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(title="Dataset", frameon=True)

    plt.tight_layout()
    plt.show()