"""
momentum.py
Detects if an altcoin is trending, coiling, or overextended.
Uses 7-day return z-score relative to rolling 90-day history of 7-day returns.
"""

import numpy as np
import pandas as pd


def compute_momentum(prices: pd.Series) -> dict:
    """
    Compute momentum state of the altcoin.

    Args:
        prices: Daily closing price series

    Returns:
        {
            return_7d: float,         # Current 7-day return %
            avg_7d_return: float,     # Rolling average of 7d returns over past 90 days
            z_score: float,           # How unusual the current 7d move is
            state: str,               # EXTENDED_UP | TRENDING | COILING | EXTENDED_DOWN
            prediction_adjustment: float,  # Multiplier adjustment on predicted move
            description: str
        }
    """
    prices = prices.dropna()
    if len(prices) < 14:
        return _neutral_momentum()

    # Current 7-day return
    ret_7d = float((prices.iloc[-1] / prices.iloc[-8]) - 1) * 100 if len(prices) >= 8 else 0.0

    # Rolling 7-day returns over last 90 days
    if len(prices) >= 14:
        rolling_7d = []
        lookback = min(90, len(prices) - 7)
        for i in range(lookback):
            idx = -(lookback - i + 7)
            end_idx = -(lookback - i) if (lookback - i) > 0 else None
            try:
                r = (prices.iloc[end_idx] / prices.iloc[idx] - 1) * 100
                rolling_7d.append(r)
            except (IndexError, ZeroDivisionError):
                pass
    else:
        rolling_7d = []

    if len(rolling_7d) < 5:
        return _neutral_momentum()

    avg_7d  = float(np.mean(rolling_7d))
    std_7d  = float(np.std(rolling_7d))
    z_score = (ret_7d - avg_7d) / std_7d if std_7d > 0 else 0.0

    # Classify and compute adjustment
    if z_score > 2.0:
        state = "EXTENDED_UP"
        # Overbought: reduce predicted upside by up to 20%
        adjustment = max(0.80, 1.0 - (z_score - 2.0) * 0.05)
        description = (
            f"Altcoin is extended to the upside (z={z_score:.1f}, +{ret_7d:.1f}% in 7 days). "
            "Predicted upside adjusted down — mean reversion risk elevated."
        )
    elif z_score < -2.0:
        state = "EXTENDED_DOWN"
        # Oversold: slightly increase predicted move (mean reversion tailwind)
        adjustment = min(1.20, 1.0 + (abs(z_score) - 2.0) * 0.05)
        description = (
            f"Altcoin is extended to the downside (z={z_score:.1f}, {ret_7d:.1f}% in 7 days). "
            "Predicted move adjusted up slightly — oversold bounce risk elevated."
        )
    elif -0.5 <= z_score <= 0.5:
        state = "COILING"
        # Flat/coiling: coin may be compressing for a bigger move
        adjustment = 1.05
        description = (
            f"Altcoin is flat/coiling (z={z_score:.1f}, {ret_7d:.1f}% in 7 days). "
            "Compressed price action often precedes amplified BTC follow-through."
        )
    else:
        state = "TRENDING"
        adjustment = 1.0
        description = (
            f"Altcoin in normal trending state (z={z_score:.1f}, {ret_7d:.1f}% in 7 days). "
            "No momentum adjustment applied."
        )

    return {
        "return_7d":              round(ret_7d, 2),
        "avg_7d_return":          round(avg_7d, 2),
        "z_score":                round(z_score, 2),
        "state":                  state,
        "prediction_adjustment":  round(adjustment, 4),
        "description":            description,
    }


def _neutral_momentum() -> dict:
    return {
        "return_7d": None, "avg_7d_return": None, "z_score": None,
        "state": "TRENDING",
        "prediction_adjustment": 1.0,
        "description": "Insufficient price history for momentum calculation.",
    }
