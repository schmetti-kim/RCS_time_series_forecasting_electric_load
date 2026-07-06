"""
tests.py — Statistical tests.
"""

import numpy as np
from scipy.stats import norm
from config import PREDICT_COLUMN

def diebold_mariano_test(
    ground_truth,
    pred_df1,
    pred_df2,
    bandwidth=None,
    prediction_col=PREDICT_COLUMN,
    loss="mse"
):
    """
    Diebold-Mariano test comparing two forecasting models using one aggregated loss per forecast origin. 
    Serial dependence in the sequence of loss differentials is accounted for using 
    a Newey-West heteroskedasticity and autocorrelation consistent (HAC) estimator.

    One aggregated loss value is computed for each forecast origin (day). 
    The Diebold-Mariano test is then applied to the resulting sequence of daily loss differentials.

    Parameters
    ----------
    ground_truth : ndarray, shape (n_days, 24)
        Ground-truth observations.

    pred_df1, pred_df2 : DataFrame
        Prediction dataframes.

    prediction_col : str
        Name of prediction column.

    loss : {"mse", "mae"}

    bandwidth : int or None, optional
        Lag truncation parameter used in the Newey-West HAC estimator for the
        long-run variance of the loss differential sequence. If None, the
        bandwidth is selected automatically using the Newey-West rule of thumb.

    Returns
    -------
    dict
    """

    y = np.asarray(ground_truth)

    p1 = pred_df1[prediction_col].to_numpy().reshape(y.shape)
    p2 = pred_df2[prediction_col].to_numpy().reshape(y.shape)

    if y.shape != p1.shape or y.shape != p2.shape:
        raise ValueError("Ground truth and prediction shapes do not match.")

    # ----- one loss per forecast origin -----

    if loss == "mse":
        loss1 = np.mean((y - p1) ** 2, axis=1)
        loss2 = np.mean((y - p2) ** 2, axis=1)

    elif loss == "mae":
        loss1 = np.mean(np.abs(y - p1), axis=1)
        loss2 = np.mean(np.abs(y - p2), axis=1)

    else:
        raise ValueError("loss must be 'mse' or 'mae'.")

    # Loss differential for each forecast origin (day).
    # Negative values favor Model 1; positive values favor Model 2.
    d = loss1 - loss2

    n = len(d)
    if n < 2:
        raise ValueError("Need at least two forecast origins.")

    # ----- HAC bandwidth -----
    # If no bandwidth is provided, use the Newey-West (1994)
    # rule-of-thumb for Bartlett-kernel HAC estimation.
    if bandwidth is None:
        # Newey-West (1994) automatic rule-of-thumb
        bandwidth = int(np.floor(4 * (n / 100) ** (2 / 9)))
    else:
        bandwidth = int(bandwidth)
        if bandwidth < 0:
            raise ValueError("bandwidth must be nonnegative.")

    d_bar = np.mean(d)

    # ----- Newey-West long-run variance -----
    # Estimate the long-run variance of the loss differential
    # sequence using the Newey-West HAC estimator.
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0

    max_lag = min(bandwidth, n - 1)

    # Add weighted sample autocovariances to estimate the
    # long-run variance using the Bartlett kernel.
    for lag in range(1, max_lag + 1):

        gamma = np.mean(
            (d[:-lag] - d_bar) *
            (d[lag:] - d_bar)
        )

        weight = 1 - lag / (max_lag + 1)

        lrv += 2 * weight * gamma

    if lrv <= 0:
        raise ValueError(
            f"Estimated long-run variance is non-positive ({lrv:.6e})."
        )

    # Diebold-Mariano statistic based on the HAC estimate of the
    # long-run variance of the loss differential sequence.
    dm_stat = d_bar / np.sqrt(lrv / n)

    # No Harvey-Leybourne-Newbold correction:
    # DM test is applied to aggregated daily losses (one loss per forecast origin),
    # with serial dependence handled by the Newey-West HAC variance estimator.
    # correction = np.sqrt(
    #     (n + 1 - 2 * h + h * (h - 1) / n) / n
    # )

    # dm_stat *= correction

    # Under the null hypothesis of equal predictive accuracy,
    # the statistic is asymptotically standard normal.
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return {
        "DM statistic": float(np.round(dm_stat, 5)),
        "p-value": float(p_value),
        "mean_loss_difference": float(np.round(d_bar, 5)),
        "winner": (
            "Model 2" if d_bar > 0
            else "Model 1" if d_bar < 0
            else "Tie"
        ),
        "n_forecasts": n,
        "bandwidth": max_lag
    }