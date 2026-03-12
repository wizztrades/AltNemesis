"""
app.py
Flask entry point. Routes: /predict, /coins, /health
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from engine.predictor        import predict
from engine.data_fetcher     import search_coins

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Rate limiter — protects CoinGecko usage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour", "30 per minute"],
    storage_uri="memory://",
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "altcoin-beta-calculator"})


@app.route("/coins")
@limiter.limit("60 per minute")
def coins():
    """Search coins by name or symbol."""
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return jsonify({"results": []})
    try:
        results = search_coins(query)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e), "results": []}), 500


@app.route("/predict")
@limiter.limit("20 per minute")
def predict_route():
    """
    Main prediction endpoint.
    Query params:
        coin    : CoinGecko coin ID (e.g. 'solana')
        btc_move: Expected BTC move in % (e.g. 3.0 or -2.5)
    """
    coin_id  = request.args.get("coin", "").strip().lower()
    btc_move = request.args.get("btc_move", "")

    if not coin_id:
        return jsonify({"success": False, "error": "Missing 'coin' parameter"}), 400

    try:
        btc_move_pct = float(btc_move)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid 'btc_move' — must be a number"}), 400

    if abs(btc_move_pct) > 50:
        return jsonify({"success": False, "error": "BTC move > 50% is not a realistic input"}), 400

    try:
        result = predict(coin_id, btc_move_pct)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500




# ── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"success": False, "error": "Rate limit exceeded. Please wait before trying again."}), 429


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ── CORS headers ─────────────────────────────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
