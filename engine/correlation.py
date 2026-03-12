"""
correlation.py
Rolling correlation health monitor.
R2 gate: determines confidence level and blocks predictions on decorrelated assets.
"""

import numpy as np
import pandas as pd


# Confidence thresholds
R2_HIGH   = 0.65
R2_MEDIUM = 0.45


def compute_correlation(returns_df: pd.DataFrame, window: int = 30) -> dict:
    """
    Compute rolling Pearson correlation and R2 between altcoin and BTC.

    Args:
        returns_df: DataFrame with columns btc_ret, alt_ret
        window: Rolling window in days

    Returns:
        {
            pearson_r: float,
            r2: float,
            confidence_gate: str,   # HIGH | MEDIUM | LOW | BLOCKED
            window: int,
            is_negative: bool,
            description: str
        }
    """
    if returns_df is None or len(returns_df) < 10:
        return _insufficient_data()

    df = returns_df.dropna().tail(window)
    if len(df) < 10:
        return _insufficient_data()

    x = df["btc_ret"].values
    y = df["alt_ret"].values

    # Pearson correlation
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return _insufficient_data()

    pearson_r = float(np.corrcoef(x, y)[0, 1])

    if np.isnan(pearson_r):
        return _insufficient_data()

    r2 = pearson_r ** 2

    is_negative = pearson_r < -0.1

    if is_negative:
        confidence_gate = "BLOCKED"
        description = (
            f"Negative correlation detected (r={pearson_r:.2f}). "
            "This altcoin is moving inversely to BTC. Prediction blocked."
        )
    elif r2 >= R2_HIGH:
        confidence_gate = "HIGH"
        description = (
            f"Strong BTC correlation (R²={r2:.2f}). "
            "Prediction confidence is high."
        )
    elif r2 >= R2_MEDIUM:
        confidence_gate = "MEDIUM"
        description = (
            f"Moderate BTC correlation (R²={r2:.2f}). "
            "Prediction is directionally reliable but magnitude has wider error."
        )
    else:
        confidence_gate = "LOW"
        description = (
            f"Weak BTC correlation (R²={r2:.2f}). "
            "This altcoin is currently decoupled from BTC. "
            "Prediction is unreliable — likely driven by coin-specific factors."
        )

    return {
        "pearson_r":       round(pearson_r, 4),
        "r2":              round(r2, 4),
        "confidence_gate": confidence_gate,
        "window":          window,
        "is_negative":     is_negative,
        "description":     description,
    }


def _insufficient_data() -> dict:
    return {
        "pearson_r":       None,
        "r2":              None,
        "confidence_gate": "LOW",
        "window":          0,
        "is_negative":     False,
        "description":     "Insufficient data for correlation calculation.",
    }
