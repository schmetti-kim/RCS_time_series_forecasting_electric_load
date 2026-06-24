"""
metrics.py — Evaluation metrics as required by the Exposé (Nti et al. 2020).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (MW)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    m = mape(y_true, y_pred)
    r = rmse(y_true, y_pred)
    print(f"  [{label}]  MAPE = {m:.4f}%   RMSE = {r:.4f} MW")
    return {"scenario": label, "MAPE_%": round(m, 6), "RMSE_MW": round(r, 6)}


def save_results(results: list[dict], results_dir: Path, dataset: str = "Panama"):
    """Save CSV + JSON (JSON used by teammates for RQ2 cross-dataset merge)."""
    df = pd.DataFrame(results)

    csv_path  = results_dir / f"{dataset.lower()}_results.csv"
    json_path = results_dir / f"{dataset.lower()}_results_rq2.json"

    df.to_csv(csv_path, index=False)

    # JSON export with dataset label — team merges all 3 datasets here
    df.insert(0, "dataset", dataset)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\n[saved]  {csv_path}")
    print(f"[saved]  {json_path}  ← share with Kim & Saxena for RQ2")
    return df