"""
dominance.py
BTC dominance trend signal. Computes 7-day slope and returns a prediction multiplier.
Rising dominance = alts underperform. Falling dominance = altseason.
"""

import numpy as np
import pandas as pd
from engine.data_fetcher import get_btc_dominance


def compute_dominance_signal(dominance_df: pd.DataFrame = None) -> dict:
    """
    Compute BTC dominance signal and multiplier.

    Returns:
        {
            current_dominance: float,
            slope_7d: float,          # % per day
            direction: str,           # RISING | FALLING | FLAT
            multiplier: float,        # Applied to predicted altcoin move
            description: str
        }
    """
    if dominance_df is None:
        try:
            dominance_df = get_btc_dominance(days=60)
        except Exception:
            return _neutral_signal("Could not fetch dominance data")

    if dominance_df is None or len(dominance_df) < 8:
        return _neutral_signal("Insufficient dominance history")

    df = dominance_df.copy().sort_values("date").reset_index(drop=True)
    dom_series = df["dominance"].values

    current_dom = float(dom_series[-1])

    # 7-day linear slope via polyfit
    last_7 = dom_series[-7:]
    x = np.arange(len(last_7))
    if len(last_7) >= 3:
        slope = float(np.polyfit(x, last_7, 1)[0])
    else:
        slope = 0.0

    # Classify direction
    if slope > 0.05:
        direction = "RISING"
        # Alts underperform — negative adjustment
        # Scale: +0.1%/day slope = -3% multiplier
        multiplier = max(0.75, 1.0 - (slope * 0.3))
        description = (
            f"BTC dominance rising at +{slope:.2f}%/day. "
            "Capital flowing into BTC, away from altcoins. Bearish for alts."
        )
    elif slope < -0.05:
        direction = "FALLING"
        # Alts outperform — positive adjustment
        multiplier = min(1.25, 1.0 + (abs(slope) * 0.3))
        description = (
            f"BTC dominance falling at {slope:.2f}%/day. "
            "Altseason conditions — alts amplifying BTC moves."
        )
    else:
        direction = "FLAT"
        multiplier = 1.0
        description = f"BTC dominance flat at {current_dom:.1f}%. Neutral signal."

    return {
        "current_dominance": round(current_dom, 2),
        "slope_7d":          round(slope, 4),
        "direction":         direction,
        "multiplier":        round(multiplier, 4),
        "description":       description,
    }


def _neutral_signal(reason: str) -> dict:
    return {
        "current_dominance": None,
        "slope_7d":          0.0,
        "direction":         "FLAT",
        "multiplier":        1.0,
        "description":       reason,
    }
