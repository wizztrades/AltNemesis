"""
lag_detector.py
Detects if an altcoin historically lags BTC moves by 1-2 days.
Uses cross-correlation analysis at multiple lag offsets.
"""

import numpy as np
import pandas as pd


LAG_SIGNIFICANCE_THRESHOLD = 0.05  # lag corr must exceed lag-0 by this to be meaningful


def detect_lag(returns_df: pd.DataFrame, max_lag: int = 2) -> dict:
    """
    Compute cross-correlation at lags 0, +1, +2 days.
    Detects if alt reliably lags BTC.

    Args:
        returns_df: DataFrame with columns btc_ret, alt_ret
        max_lag: Maximum lag to check (days)

    Returns:
        {
            lag_0_corr: float,
            lag_1_corr: float,
            lag_2_corr: float,
            best_lag: int,
            is_lagging: bool,
            lag_days: int,
            description: str
        }
    """
    df = returns_df.dropna()
    if len(df) < 30:
        return _no_lag()

    btc = df["btc_ret"].values
    alt = df["alt_ret"].values

    lag_corrs = {}
    for lag in range(max_lag + 1):
        if lag == 0:
            x = btc
            y = alt
        else:
            x = btc[:-lag]
            y = alt[lag:]

        if len(x) < 20:
            lag_corrs[lag] = 0.0
            continue

        try:
            corr = float(np.corrcoef(x, y)[0, 1])
            lag_corrs[lag] = corr if not np.isnan(corr) else 0.0
        except Exception:
            lag_corrs[lag] = 0.0

    lag_0 = lag_corrs.get(0, 0.0)
    lag_1 = lag_corrs.get(1, 0.0)
    lag_2 = lag_corrs.get(2, 0.0)

    # Best lag by absolute correlation
    best_lag = max(lag_corrs, key=lambda k: abs(lag_corrs[k]))

    # Is lagging? Only flag if a non-zero lag is meaningfully better
    is_lagging = (
        best_lag > 0 and
        abs(lag_corrs[best_lag]) > abs(lag_0) + LAG_SIGNIFICANCE_THRESHOLD
    )

    if is_lagging:
        description = (
            f"This coin shows stronger correlation to BTC at lag +{best_lag} day(s) "
            f"(r={lag_corrs[best_lag]:.2f}) vs lag 0 (r={lag_0:.2f}). "
            f"It historically follows BTC moves with a ~{best_lag} day delay."
        )
    else:
        description = (
            f"No significant lag detected. Correlation is highest at lag 0 (r={lag_0:.2f}). "
            "This coin moves with BTC simultaneously."
        )

    return {
        "lag_0_corr": round(lag_0, 4),
        "lag_1_corr": round(lag_1, 4),
        "lag_2_corr": round(lag_2, 4),
        "best_lag":   best_lag,
        "is_lagging": is_lagging,
        "lag_days":   best_lag if is_lagging else 0,
        "description": description,
    }


def _no_lag() -> dict:
    return {
        "lag_0_corr": None, "lag_1_corr": None, "lag_2_corr": None,
        "best_lag": 0, "is_lagging": False, "lag_days": 0,
        "description": "Insufficient data for lag analysis.",
    }
