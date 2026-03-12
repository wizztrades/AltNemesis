"""
predictor.py
Master orchestrator. Runs all 7 modules in sequence and assembles
the final prediction with confidence tier and full signal breakdown.
"""

import numpy as np
import pandas as pd
import traceback

from engine.data_fetcher      import get_aligned_returns, get_market_chart, get_btc_dominance
from engine.regime_detector   import detect_regime
from engine.beta_engine       import compute_all_windows
from engine.dominance         import compute_dominance_signal
from engine.correlation       import compute_correlation
from engine.volatility        import compute_volatility_signal
from engine.lag_detector      import detect_lag
from engine.momentum          import compute_momentum


# Base range half-width per confidence tier (as % of predicted move)
BASE_RANGE_PCT = {
    "HIGH":   0.60,   # ±60% of predicted move (widened from 0.30)
    "MEDIUM": 0.80,   # ±80%                   (widened from 0.50)
    "LOW":    1.20,   # ±120%                  (widened from 0.80)
    "BLOCKED": 1.0,
}


def predict(coin_id: str, btc_move_pct: float) -> dict:
    """
    Full adaptive prediction pipeline.

    Args:
        coin_id: CoinGecko coin ID (e.g. 'solana', 'ethereum')
        btc_move_pct: Expected BTC move in % (e.g. 3.0 for +3%, -2.5 for -2.5%)

    Returns:
        Full prediction dict with all signals and metadata.
    """
    warnings = []
    errors   = []

    # ── 1. Fetch data ─────────────────────────────────────────────────────
    try:
        returns_df = get_aligned_returns(coin_id, days=365)
        prices_df  = get_market_chart(coin_id, days=365)
        btc_prices = get_market_chart("bitcoin", days=365)
    except Exception as e:
        return _error_response(coin_id, btc_move_pct, f"Data fetch failed: {e}")

    if len(returns_df) < 30:
        return _error_response(coin_id, btc_move_pct, "Insufficient historical data (need 30+ days)")

    alt_prices = prices_df.set_index("date")["price"] if "date" in prices_df.columns else prices_df["price"]

    # ── 2. Regime Detection ───────────────────────────────────────────────
    try:
        regime = detect_regime(btc_prices)
    except Exception as e:
        regime = {"regime": "SIDEWAYS", "label": "Unknown", "description": str(e)}
        errors.append(f"Regime detection failed: {e}")

    # ── 3. Beta Computation ───────────────────────────────────────────────
    try:
        beta_data = compute_all_windows(returns_df)
    except Exception as e:
        return _error_response(coin_id, btc_move_pct, f"Beta calculation failed: {e}")

    # Select appropriate beta based on direction
    direction = "up" if btc_move_pct >= 0 else "down"
    if direction == "up":
        selected_beta = beta_data["consensus_beta_up"]
    else:
        selected_beta = beta_data["consensus_beta_down"]

    # Fallback to all-direction consensus
    if selected_beta is None:
        selected_beta = beta_data.get("consensus_beta_up") or beta_data.get("consensus_beta_down")
        if selected_beta is None:
            return _error_response(coin_id, btc_move_pct, "Could not compute beta — not enough data")

    # ── 4. Dominance Signal ───────────────────────────────────────────────
    try:
        dom_df = get_btc_dominance(days=60)
        dominance = compute_dominance_signal(dom_df)
    except Exception as e:
        dominance = {"multiplier": 1.0, "direction": "FLAT", "description": str(e)}
        warnings.append("Dominance signal unavailable — using neutral")

    # ── 5. Correlation Health ─────────────────────────────────────────────
    try:
        correlation = compute_correlation(returns_df, window=30)
    except Exception as e:
        correlation = {"confidence_gate": "LOW", "r2": None, "description": str(e)}
        warnings.append("Correlation check failed")

    if correlation["confidence_gate"] == "BLOCKED":
        return _blocked_response(coin_id, btc_move_pct, correlation, regime)

    # ── 6. Volatility Scaling ─────────────────────────────────────────────
    try:
        vol_signal = compute_volatility_signal(returns_df["alt_ret"])
    except Exception as e:
        vol_signal = {"range_multiplier": 1.0, "regime": "NORMAL", "description": str(e)}
        warnings.append("Volatility calculation failed — using neutral range")

    # ── 7. Lag Detection ─────────────────────────────────────────────────
    try:
        lag = detect_lag(returns_df)
        if lag["is_lagging"]:
            warnings.append(
                f"⏱ This coin historically lags BTC by ~{lag['lag_days']} day(s). "
                "If BTC has already moved, this coin may be playing catch-up."
            )
    except Exception as e:
        lag = {"is_lagging": False, "lag_days": 0, "description": str(e)}

    # ── 8. Momentum State ─────────────────────────────────────────────────
    try:
        momentum = compute_momentum(alt_prices)
    except Exception as e:
        momentum = {"prediction_adjustment": 1.0, "state": "TRENDING", "description": str(e)}
        warnings.append("Momentum analysis failed — no adjustment applied")

    # ── 9. Regime Adjustments on Beta ────────────────────────────────────
    regime_code = regime.get("regime", "SIDEWAYS")
    if regime_code == "SHOCK":
        warnings.append(
            "⚡ Volatility shock detected. Beta relationships break down in shock conditions. "
            "Use this prediction with extreme caution."
        )
    if regime_code == "BEAR" and direction == "up":
        warnings.append(
            "🐻 Bear regime: alts often struggle to hold gains even when BTC bounces."
        )

    # ── 10. Assemble Final Prediction ─────────────────────────────────────
    # Core: beta × BTC move
    raw_predicted = selected_beta * btc_move_pct

    # Stablecoin / near-zero beta guard
    if abs(raw_predicted) < 0.15:
        return {
            "success": False,
            "coin_id": coin_id,
            "btc_move_input": btc_move_pct,
            "confidence": "BLOCKED",
            "correlation_gate": "BLOCKED",
            "r2": correlation.get("r2"),
            "regime": regime.get("label"),
            "error": "Prediction blocked: near-zero beta. This may be a stablecoin or fully decorrelated asset.",
            "warnings": [
                "Prediction blocked: beta is effectively zero. This asset does not move with BTC — it may be a stablecoin or uncorrelated token."
            ],
        }

    # Apply dominance multiplier — halved weight so it nudges rather than overrides
    dom_multiplier = 1.0 + (dominance["multiplier"] - 1.0) * 0.5
    adjusted_predicted = raw_predicted * dom_multiplier

    # Apply momentum adjustment — halved weight
    mom_adjustment = 1.0 + (momentum["prediction_adjustment"] - 1.0) * 0.5
    adjusted_predicted = adjusted_predicted * mom_adjustment

    # Confidence tier
    confidence = _determine_confidence(
        correlation=correlation,
        beta_data=beta_data,
        regime_code=regime_code,
        vol_regime=vol_signal.get("regime", "NORMAL"),
    )

    # Prediction range
    base_half_width = abs(adjusted_predicted) * BASE_RANGE_PCT.get(confidence, 0.5)
    half_width = base_half_width * vol_signal.get("range_multiplier", 1.0)
    half_width = max(half_width, 0.5)  # Minimum ±0.5% range

    range_low  = round(adjusted_predicted - half_width, 2)
    range_high = round(adjusted_predicted + half_width, 2)

    # ── 11. Build scatter data (last 90 days) ─────────────────────────────
    scatter_data = _build_scatter(returns_df.tail(90), selected_beta)

    return {
        # Core output
        "coin_id":         coin_id,
        "btc_move_input":  btc_move_pct,
        "predicted_move":  round(adjusted_predicted, 2),
        "range_low":       range_low,
        "range_high":      range_high,
        "confidence":      confidence,

        # Regime
        "regime":          regime.get("label", "Unknown"),
        "regime_code":     regime_code,
        "regime_description": regime.get("description", ""),
        "btc_ma50":        regime.get("ma50"),
        "btc_ma200":       regime.get("ma200"),
        "btc_price":       regime.get("current_price"),

        # Beta
        "beta_up":             beta_data.get("consensus_beta_up"),
        "beta_down":           beta_data.get("consensus_beta_down"),
        "beta_selected":       round(selected_beta, 4),
        "beta_agreement":      beta_data.get("agreement"),
        "agreeing_windows":    beta_data.get("agreeing_windows", []),
        "windows_detail":      _format_windows(beta_data.get("windows", {})),

        # Correlation
        "r2":              correlation.get("r2"),
        "pearson_r":       correlation.get("pearson_r"),
        "correlation_gate": correlation.get("confidence_gate"),

        # Dominance
        "dominance_direction":  dominance.get("direction"),
        "dominance_multiplier": dominance.get("multiplier"),
        "dominance_value":      dominance.get("current_dominance"),
        "dominance_description": dominance.get("description"),

        # Volatility
        "vol_ratio":        vol_signal.get("vol_ratio"),
        "vol_regime":       vol_signal.get("regime"),
        "vol_description":  vol_signal.get("description"),
        "vol_14d_ann":      vol_signal.get("vol_14d_ann"),

        # Lag
        "lag_days":         lag.get("lag_days", 0),
        "is_lagging":       lag.get("is_lagging", False),
        "lag_description":  lag.get("description"),

        # Momentum
        "momentum_state":   momentum.get("state"),
        "momentum_z":       momentum.get("z_score"),
        "momentum_7d_ret":  momentum.get("return_7d"),
        "momentum_description": momentum.get("description"),

        # Meta
        "warnings": warnings,
        "errors":   errors,
        "scatter":  scatter_data,
        "success":  True,
    }


def _determine_confidence(correlation, beta_data, regime_code, vol_regime):
    """Map signal quality to HIGH / MEDIUM / LOW."""
    r2 = correlation.get("r2") or 0.0
    agreement = beta_data.get("agreement", "LOW")
    gate = correlation.get("confidence_gate", "LOW")

    if regime_code == "SHOCK":
        return "LOW"

    if gate == "HIGH" and agreement == "HIGH" and vol_regime != "HIGH":
        return "HIGH"
    elif gate in ("HIGH", "MEDIUM") and agreement in ("HIGH", "MEDIUM"):
        return "MEDIUM"
    else:
        return "LOW"


def _format_windows(windows: dict) -> list:
    out = []
    for w, d in windows.items():
        out.append({
            "window":     w,
            "beta_up":    d.get("beta_up"),
            "beta_down":  d.get("beta_down"),
            "beta_all":   d.get("beta_all"),
            "r2":         d.get("r2_all"),
            "n":          d.get("n_all"),
        })
    return out


def _build_scatter(returns_df: pd.DataFrame, beta: float) -> dict:
    """Build scatter plot data for frontend."""
    df = returns_df.dropna()
    points = [
        {"x": round(r["btc_ret"] * 100, 3), "y": round(r["alt_ret"] * 100, 3)}
        for _, r in df.iterrows()
    ]
    # Regression line points
    btc_vals = df["btc_ret"].values * 100
    if len(btc_vals) > 0:
        x_min, x_max = float(btc_vals.min()), float(btc_vals.max())
        line = [
            {"x": round(x_min, 2), "y": round(beta * x_min, 2)},
            {"x": round(x_max, 2), "y": round(beta * x_max, 2)},
        ]
    else:
        line = []
    return {"points": points, "regression_line": line}


def _error_response(coin_id, btc_move, message):
    return {
        "success": False, "coin_id": coin_id,
        "btc_move_input": btc_move, "error": message,
        "confidence": "LOW", "warnings": [message],
    }


def _blocked_response(coin_id, btc_move, correlation, regime):
    return {
        "success": False,
        "coin_id": coin_id,
        "btc_move_input": btc_move,
        "confidence": "BLOCKED",
        "correlation_gate": "BLOCKED",
        "r2": correlation.get("r2"),
        "regime": regime.get("label"),
        "error": correlation.get("description"),
        "warnings": [
            "Prediction blocked: this altcoin has negative or near-zero correlation with BTC. "
            "It is currently trading on its own narrative."
        ],
        "success": False,
    }
