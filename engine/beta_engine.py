"""
beta_engine.py
Computes asymmetric (up/down) EWMA-weighted beta across multiple windows.
This is the statistical core of the prediction engine.
"""

import numpy as np
import pandas as pd
from scipy import stats


WINDOWS = [30, 90, 180]
EWMA_SPAN = {30: 15, 90: 45, 180: 90}  # EWMA span = half the window


def _ewma_weights(n: int, span: int) -> np.ndarray:
    """
    Generate exponentially decaying weights.
    More recent observations receive higher weight.
    """
    alpha = 2.0 / (span + 1)
    indices = np.arange(n - 1, -1, -1)  # Oldest = 0, newest = n-1
    weights = (1 - alpha) ** indices
    return weights / weights.sum()


def _weighted_ols(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    """
    Weighted OLS regression: y = beta * x + alpha
    Returns beta, alpha, r_squared, std_err
    """
    if len(x) < 5:
        return {"beta": None, "alpha": None, "r2": None, "std_err": None, "n": len(x)}

    w = weights / weights.sum()

    # Weighted means
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)

    # Weighted covariance and variance
    cov_xy = np.average((x - x_mean) * (y - y_mean), weights=w)
    var_x  = np.average((x - x_mean) ** 2, weights=w)

    if var_x < 1e-10:
        return {"beta": None, "alpha": None, "r2": None, "std_err": None, "n": len(x)}

    beta  = cov_xy / var_x
    alpha = y_mean - beta * x_mean

    # R-squared (weighted)
    y_pred = alpha + beta * x
    ss_res = np.average((y - y_pred) ** 2, weights=w)
    ss_tot = np.average((y - y_mean) ** 2, weights=w)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    r2 = max(0.0, r2)

    # Standard error
    residuals = y - y_pred
    mse = np.average(residuals ** 2, weights=w)
    std_err = np.sqrt(mse / var_x) if var_x > 0 else 0.0

    return {
        "beta":    round(float(beta), 4),
        "alpha":   round(float(alpha), 6),
        "r2":      round(float(r2), 4),
        "std_err": round(float(std_err), 4),
        "n":       len(x)
    }


def compute_beta_for_window(returns_df: pd.DataFrame, window: int) -> dict:
    """
    Compute asymmetric EWMA beta for a given lookback window.

    Args:
        returns_df: DataFrame with columns btc_ret, alt_ret
        window: lookback in days

    Returns:
        {
            window: int,
            beta_up: float,       # beta on BTC positive days
            beta_down: float,     # beta on BTC negative days
            beta_all: float,      # overall beta
            r2_up: float,
            r2_down: float,
            r2_all: float,
            n_up: int,
            n_down: int,
            n_all: int
        }
    """
    df = returns_df.tail(window).copy()
    if len(df) < 20:
        return _empty_window(window)

    x_all = df["btc_ret"].values
    y_all = df["alt_ret"].values
    w_all = _ewma_weights(len(df), EWMA_SPAN[window])

    # All-direction beta
    all_result = _weighted_ols(x_all, y_all, w_all)

    # Upside: days where BTC was positive
    up_mask = x_all > 0
    x_up = x_all[up_mask]
    y_up = y_all[up_mask]
    if len(x_up) >= 5:
        w_up = _ewma_weights(len(x_up), max(5, EWMA_SPAN[window] // 2))
        up_result = _weighted_ols(x_up, y_up, w_up)
    else:
        up_result = {"beta": all_result["beta"], "r2": all_result["r2"], "n": len(x_up)}

    # Downside: days where BTC was negative
    down_mask = x_all < 0
    x_down = x_all[down_mask]
    y_down = y_all[down_mask]
    if len(x_down) >= 5:
        w_down = _ewma_weights(len(x_down), max(5, EWMA_SPAN[window] // 2))
        down_result = _weighted_ols(x_down, y_down, w_down)
    else:
        down_result = {"beta": all_result["beta"], "r2": all_result["r2"], "n": len(x_down)}

    return {
        "window":    window,
        "beta_up":   up_result.get("beta"),
        "beta_down": down_result.get("beta"),
        "beta_all":  all_result.get("beta"),
        "r2_up":     up_result.get("r2"),
        "r2_down":   down_result.get("r2"),
        "r2_all":    all_result.get("r2"),
        "n_up":      int(up_mask.sum()),
        "n_down":    int(down_mask.sum()),
        "n_all":     len(df),
    }


def _empty_window(window: int) -> dict:
    return {
        "window": window, "beta_up": None, "beta_down": None, "beta_all": None,
        "r2_up": None, "r2_down": None, "r2_all": None,
        "n_up": 0, "n_down": 0, "n_all": 0
    }


def compute_all_windows(returns_df: pd.DataFrame) -> dict:
    """
    Compute beta across all three windows and determine consensus.

    Returns:
        {
            windows: { 30: {...}, 90: {...}, 180: {...} },
            consensus_beta_up: float,
            consensus_beta_down: float,
            consensus_r2: float,
            agreement: str,        # HIGH | MEDIUM | LOW
            agreeing_windows: list,
            disagreeing_windows: list
        }
    """
    results = {}
    for w in WINDOWS:
        results[w] = compute_beta_for_window(returns_df, w)

    # Extract valid betas
    valid_up   = {w: r["beta_up"]   for w, r in results.items() if r["beta_up"]   is not None}
    valid_down = {w: r["beta_down"] for w, r in results.items() if r["beta_down"] is not None}
    valid_all  = {w: r["beta_all"]  for w, r in results.items() if r["beta_all"]  is not None}

    if not valid_all:
        return {
            "windows": results,
            "consensus_beta_up": None, "consensus_beta_down": None,
            "consensus_r2": None, "agreement": "LOW",
            "agreeing_windows": [], "disagreeing_windows": list(WINDOWS)
        }

    # Consensus: check directional agreement among valid betas
    up_signs   = [np.sign(b) for b in valid_up.values()]
    down_signs = [np.sign(b) for b in valid_down.values()]

    # Check if majority agree
    agreeing_windows = []
    disagreeing_windows = []

    if len(valid_all) >= 2:
        all_betas = list(valid_all.values())
        median_sign = np.sign(np.median(all_betas))
        for w, b in valid_all.items():
            if np.sign(b) == median_sign:
                agreeing_windows.append(w)
            else:
                disagreeing_windows.append(w)

    n_agree = len(agreeing_windows)
    if n_agree == 3:
        agreement = "HIGH"
    elif n_agree == 2:
        agreement = "MEDIUM"
    else:
        agreement = "LOW"

    # Consensus beta = EWMA-weighted average of agreeing windows
    # Longer windows get less weight (30d most recent = most weight)
    window_weights = {30: 0.5, 90: 0.3, 180: 0.2}

    def weighted_consensus(beta_dict, windows_to_use):
        total_w, total_beta = 0.0, 0.0
        for w in windows_to_use:
            if w in beta_dict and beta_dict[w] is not None:
                wt = window_weights.get(w, 0.33)
                total_beta += beta_dict[w] * wt
                total_w += wt
        return round(total_beta / total_w, 4) if total_w > 0 else None

    use_windows = agreeing_windows if agreeing_windows else list(valid_all.keys())

    consensus_beta_up   = weighted_consensus(valid_up, use_windows)
    consensus_beta_down = weighted_consensus(valid_down, use_windows)

    # Consensus R2 = average R2 of agreeing windows
    r2_values = [results[w]["r2_all"] for w in use_windows if results[w]["r2_all"] is not None]
    consensus_r2 = round(float(np.mean(r2_values)), 4) if r2_values else None

    return {
        "windows":             results,
        "consensus_beta_up":   consensus_beta_up,
        "consensus_beta_down": consensus_beta_down,
        "consensus_r2":        consensus_r2,
        "agreement":           agreement,
        "agreeing_windows":    agreeing_windows,
        "disagreeing_windows": disagreeing_windows,
    }
