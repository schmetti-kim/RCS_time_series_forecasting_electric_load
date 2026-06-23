"""
plots.py — Visualisation helpers for forecast results and EDA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame, timestamp_col: str, target_col: str,
             results_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    axes[0].plot(df[timestamp_col], df[target_col], lw=0.5, color="steelblue")
    axes[0].set_title("Panama — National Electricity Demand (full series)", fontsize=12)
    axes[0].set_ylabel("MW")

    zoom = df.tail(24 * 14)
    axes[1].plot(zoom[timestamp_col], zoom[target_col], lw=0.9, color="darkorange")
    axes[1].set_title("Last 2 weeks (hourly detail)", fontsize=12)
    axes[1].set_ylabel("MW")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.tight_layout()
    out = results_dir / "eda_demand.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot]  {out}")


# ── Forecast ──────────────────────────────────────────────────────────────────

def plot_forecast(y_true: np.ndarray, y_pred: np.ndarray,
                  title: str, filename: str, results_dir: Path,
                  q_lo: np.ndarray = None, q_hi: np.ndarray = None,
                  ensemble_std: np.ndarray = None):
    """
    Plot actual vs. point forecast.
    Optionally add a shaded prediction interval from either:
      - quantile band  (q_lo, q_hi)   — S1, S2, S3
      - ensemble spread (ensemble_std) — S4
    """
    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(x, y_true,  label="Actual",           color="black",     lw=1.5)
    ax.plot(x, y_pred,  label="Forecast (median)", color="steelblue", lw=1.5, ls="--")

    if q_lo is not None and q_hi is not None:
        ax.fill_between(x, q_lo, q_hi, alpha=0.25, color="steelblue", label="10–90 % PI")

    if ensemble_std is not None:
        ax.fill_between(x,
                        y_pred - 2 * ensemble_std,
                        y_pred + 2 * ensemble_std,
                        alpha=0.25, color="purple", label="±2σ ensemble spread")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.legend()
    plt.tight_layout()

    out = results_dir / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot]  {out}")


# ── STL decomposition ─────────────────────────────────────────────────────────

def plot_stl(trend, seasonal, residual, original, results_dir: Path,
             n_days: int = 30):
    n = 24 * n_days
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    for ax, data, label, color in zip(
        axes,
        [original[-n:], trend[-n:], seasonal[-n:], residual[-n:]],
        ["Original", "Trend", "Seasonal (24 h)", "Residual"],
        ["black", "steelblue", "darkorange", "green"],
    ):
        ax.plot(data, color=color, lw=0.8)
        ax.set_ylabel(label)
    axes[0].set_title(f"STL Decomposition — last {n_days} days", fontsize=12)
    plt.tight_layout()
    out = results_dir / "s3b_stl_decomposition.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot]  {out}")


# ── Summary bar chart ─────────────────────────────────────────────────────────

def plot_summary(results_df: pd.DataFrame, results_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = sns.color_palette("muted", len(results_df))

    for ax, metric, unit in zip(axes,
                                 ["MAPE_%", "RMSE_MW"],
                                 ["%", "MW"]):
        bars = ax.bar(results_df["scenario"], results_df[metric],
                      color=colors, edgecolor="white")
        ax.set_title(f"{metric.split('_')[0]} by Scenario — Panama", fontsize=12)
        ax.set_ylabel(f"{metric.split('_')[0]} ({unit})")
        ax.tick_params(axis="x", rotation=35)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Chronos-2 Adaptation Strategies — Panama", fontsize=13, y=1.01)
    plt.tight_layout()
    out = results_dir / "scenario_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  {out}")