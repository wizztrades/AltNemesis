# AltNemesis

Adaptive regime-aware crypto prediction engine. Predicts altcoin moves given a BTC move %.

## Stack
- Python 3.11+ / Flask / Gunicorn
- NumPy, SciPy, Pandas (statistics)
- CoinGecko Public API 
- SQLite (caching)
- Tailwind CSS + Chart.js (frontend)
- FluxCloud


---

## Local Setup

```bash
# 1. Clone
git clone <your-repo-url>
cd altcoin-calculator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

Open http://localhost:5000

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Calculator UI |
| `/backtest-page` | GET | Backtester UI |
| `/predict?coin=solana&btc_move=3.0` | GET | Run prediction |
| `/coins?q=sol` | GET | Search coins |
| `/backtest?coin=solana&days=180&min_btc_move=1.0` | GET | Run backtest |
| `/health` | GET | Health check |

---

## Architecture

```
engine/
  data_fetcher.py     # CoinGecko API + SQLite cache (6hr TTL)
  regime_detector.py  # Bull / Bear / Sideways / Shock classification
  beta_engine.py      # Asymmetric EWMA beta (30/90/180d windows)
  dominance.py        # BTC dominance signal + multiplier
  correlation.py      # Rolling Pearson R² gate
  volatility.py       # Vol ratio → prediction range width
  lag_detector.py     # Cross-correlation lag analysis
  momentum.py         # 7d return z-score
  predictor.py        # Master orchestrator

backtest/
  backtest_engine.py  # Walk-forward simulation (no lookahead bias)
```

---

## Confidence Tiers

| Tier | Criteria |
|---|---|
| 🟢 HIGH | R² > 0.65 AND all 3 windows agree AND not shock regime |
| 🟡 MEDIUM | R² 0.45–0.65 OR 2/3 windows agree |
| 🔴 LOW | R² < 0.45 OR windows disagree OR shock regime |
| ⛔ BLOCKED | Negative correlation — prediction suppressed |
