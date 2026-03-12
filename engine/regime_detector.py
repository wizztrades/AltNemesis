"""
regime_detector.py
Classifies current BTC market regime: Bull / Bear / Sideways / Shock
Uses: MA crossovers + ATR ratio. All computed from free CoinGecko data.
"""

import numpy as np
import pandas as pd
from engine.data_fetcher import get_market_chart


REGIMES = {
    "BULL":     "Bull Trending",
    "BEAR":     "Bear Trending",
    "SIDEWAYS": "Sideways / Accumulation",
    "SHOCK":    "Volatility Shock",
}


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Approximate ATR using daily price range from OHLC.
    If only close prices available, use rolling std as proxy.
    """
    if "high" in df.columns and "low" in df.columns:
        tr = df["high"] - df["low"]
    else:
        # Proxy: rolling std of daily returns * price
        tr = df["price"].pct_change().abs() * df["price"]
    return tr.rolling(window).mean()


def detect_regime(btc_prices: pd.DataFrame = None) -> dict:
    """
    Detect current BTC market regime.

    Args:
        btc_prices: Optional pre-fetched DataFrame with 'price' column indexed by date.
                    If None, fetches fresh data.

    Returns:
        {
            regime: str,          # BULL | BEAR | SIDEWAYS | SHOCK
            label: str,           # Human readable
            ma50: float,
            ma200: float,
            current_price: float,
            atr_ratio: float,     # current 14d ATR / 90d avg ATR
            above_ma50: bool,
            above_ma200: bool,
            description: str
        }
    """
    if btc_prices is None:
        df = get_market_chart("bitcoin", days=365)
    else:
        df = btc_prices.copy()
        if "date" in df.columns:
            df = df.set_index("date") if df.index.name != "date" else df

    if "price" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    prices = df["price"] if "price" in df.columns else df.iloc[:, 0]
    prices = prices.dropna()

    if len(prices) < 50:
        return {
            "regime": "SIDEWAYS",
            "label": REGIMES["SIDEWAYS"],
            "description": "Insufficient data for regime detection",
            "ma50": None, "ma200": None,
            "current_price": float(prices.iloc[-1]) if len(prices) > 0 else 0,
            "atr_ratio": 1.0,
            "above_ma50": False, "above_ma200": False,
        }

    current_price = float(prices.iloc[-1])
    ma50  = float(prices.rolling(50).mean().iloc[-1])
    ma200 = float(prices.rolling(200).mean().iloc[-1]) if len(prices) >= 200 else float(prices.rolling(len(prices)).mean().iloc[-1])

    # ATR proxy: rolling std of daily returns
    daily_returns = prices.pct_change().dropna()
    atr_14  = float(daily_returns.rolling(14).std().iloc[-1])
    atr_90  = float(daily_returns.rolling(90).std().iloc[-1]) if len(daily_returns) >= 90 else atr_14
    atr_ratio = atr_14 / atr_90 if atr_90 > 0 else 1.0

    above_ma50  = current_price > ma50
    above_ma200 = current_price > ma200

    # ── Regime Classification ─────────────────────────────────────────────
    if atr_ratio > 1.5:
        regime = "SHOCK"
        description = (
            f"Volatility is {atr_ratio:.1f}x the 90-day average. "
            "Beta relationships are unreliable during shock conditions."
        )
    elif above_ma50 and above_ma200:
        regime = "BULL"
        description = (
            f"BTC trading above both 50d MA (${ma50:,.0f}) and 200d MA (${ma200:,.0f}). "
            "Altcoins tend to amplify BTC moves in this regime."
        )
    elif not above_ma50 and not above_ma200:
        regime = "BEAR"
        description = (
            f"BTC below both 50d MA (${ma50:,.0f}) and 200d MA (${ma200:,.0f}). "
            "Altcoins tend to bleed harder than BTC in this regime."
        )
    else:
        regime = "SIDEWAYS"
        description = (
            f"BTC in transition zone. "
            f"{'Above' if above_ma50 else 'Below'} 50d MA, "
            f"{'above' if above_ma200 else 'below'} 200d MA. "
            "Predictions carry higher uncertainty."
        )

    return {
        "regime": regime,
        "label": REGIMES[regime],
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "current_price": round(current_price, 2),
        "atr_ratio": round(atr_ratio, 3),
        "above_ma50": above_ma50,
        "above_ma200": above_ma200,
        "description": description,
    }
