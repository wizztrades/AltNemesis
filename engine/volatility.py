"""
volatility.py
Computes current vs historical volatility ratio.
Determines prediction range width: wider in high-vol, tighter in compressed-vol regimes.
"""

import numpy as np
import pandas as pd


def compute_volatility_signal(returns: pd.Series) -> dict:
    """
    Compute volatility ratio and range multiplier.

    Args:
        returns: Daily percentage return series for the altcoin

    Returns:
        {
            vol_14d: float,       # Current 14-day rolling std (annualised)
            vol_90d: float,       # 90-day rolling std (annualised)
            vol_ratio: float,     # current / historical
            range_multiplier: float,  # Applied to prediction range width
            regime: str,          # HIGH | NORMAL | LOW
            description: str
        }
    """
    returns = returns.dropna()
    if len(returns) < 15:
        return _default_signal()

    # Daily std
    vol_14d = float(returns.tail(14).std())
    vol_90d = float(returns.tail(90).std()) if len(returns) >= 90 else float(returns.std())

    if vol_90d < 1e-8:
        return _default_signal()

    vol_ratio = vol_14d / vol_90d

    # Annualised vols (for display)
    ann_14 = round(vol_14d * np.sqrt(365) * 100, 2)
    ann_90 = round(vol_90d * np.sqrt(365) * 100, 2)

    if vol_ratio > 1.5:
        regime = "HIGH"
        range_multiplier = 1.0 + (vol_ratio - 1.0) * 0.6  # Widen significantly
        range_multiplier = min(2.5, range_multiplier)
        description = (
            f"Volatility is {vol_ratio:.1f}x the 90-day average ({ann_14}% ann. vs {ann_90}% baseline). "
            "Prediction range significantly widened."
        )
    elif vol_ratio < 0.7:
        regime = "LOW"
        range_multiplier = max(0.6, vol_ratio * 0.9)
        description = (
            f"Volatility is compressed at {vol_ratio:.1f}x the 90-day average. "
            "Prediction range tightened — higher precision expected."
        )
    else:
        regime = "NORMAL"
        range_multiplier = 1.0
        description = (
            f"Volatility is normal at {vol_ratio:.1f}x the 90-day average ({ann_14}% ann.)."
        )

    return {
        "vol_14d":          round(vol_14d * 100, 4),  # as %
        "vol_90d":          round(vol_90d * 100, 4),
        "vol_14d_ann":      ann_14,
        "vol_90d_ann":      ann_90,
        "vol_ratio":        round(vol_ratio, 3),
        "range_multiplier": round(range_multiplier, 3),
        "regime":           regime,
        "description":      description,
    }


def _default_signal() -> dict:
    return {
        "vol_14d": None, "vol_90d": None,
        "vol_14d_ann": None, "vol_90d_ann": None,
        "vol_ratio": 1.0,
        "range_multiplier": 1.0,
        "regime": "NORMAL",
        "description": "Insufficient data for volatility calculation.",
    }
