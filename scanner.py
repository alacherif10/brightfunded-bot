import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from trade_engine import load_state, save_state, reset_daily_if_new_day, get_risk_status, calculate_lot_size

# ============ WATCHLIST ============
INSTRUMENTS = {
    # FX majors
    "EURUSD=X":   {"name": "EURUSD", "type": "forex",  "pip": 0.0001, "pip_value": 10.0},
    "GBPUSD=X":   {"name": "GBPUSD", "type": "forex",  "pip": 0.0001, "pip_value": 10.0},
    "USDJPY=X":   {"name": "USDJPY", "type": "forex",  "pip": 0.01,   "pip_value": 6.7},
    "USDCHF=X":   {"name": "USDCHF", "type": "forex",  "pip": 0.0001, "pip_value": 11.0},
    "USDCAD=X":   {"name": "USDCAD", "type": "forex",  "pip": 0.0001, "pip_value": 7.3},
    "AUDUSD=X":   {"name": "AUDUSD", "type": "forex",  "pip": 0.0001, "pip_value": 10.0},
    "NZDUSD=X":   {"name": "NZDUSD", "type": "forex",  "pip": 0.0001, "pip_value": 10.0},
    # FX crosses
    "EURGBP=X":   {"name": "EURGBP", "type": "forex",  "pip": 0.0001, "pip_value": 13.0},
    "EURJPY=X":   {"name": "EURJPY", "type": "forex",  "pip": 0.01,   "pip_value": 6.7},
    "GBPJPY=X":   {"name": "GBPJPY", "type": "forex",  "pip": 0.01,   "pip_value": 6.7},
    # Metals
    "GC=F":       {"name": "XAUUSD", "type": "metal",  "pip": 0.10,   "pip_value": 10.0},
    "SI=F":       {"name": "XAGUSD", "type": "metal",  "pip": 0.01,   "pip_value": 50.0},
    # Indices
    "^GSPC":      {"name": "US500",  "type": "index",  "pip": 1.0,    "pip_value": 1.0},
    "^NDX":       {"name": "US100",  "type": "index",  "pip": 1.0,    "pip_value": 1.0},
    "^GDAXI":     {"name": "GER40",  "type": "index",  "pip": 1.0,    "pip_value": 1.0},
    # Crypto
    "BTC-USD":    {"name": "BTCUSD", "type": "crypto", "pip": 1.0,    "pip_value": 1.0},
    "ETH-USD":    {"name": "ETHUSD", "type": "crypto", "pip": 0.10,   "pip_value": 1.0},
}
# ====================================


def add_indicators(df):
    df = df.copy()
    df["EMA20"]  = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]  = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Volume-based volatility filter
    df["ATR_pct"] = df["ATR"] / df["Close"]
    return df


def score_signal(df, meta):
    """
    Multi-factor signal scorer. Returns list of candidate signals with confluence score.
    """
    if len(df) < 200:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2]
    candidates = []

    trend_up   = last["EMA50"] > last["EMA200"] and last["EMA20"] > last["EMA50"]
    trend_down = last["EMA50"] < last["EMA200"] and last["EMA20"] < last["EMA50"]

    # --- STRATEGY 1: Trend pullback to EMA20 (highest quality) ---
    if trend_up:
        distance = (last["Close"] - last["EMA20"]) / last["ATR"]
        if -0.5 < distance < 0.5 and 40 < last["RSI"] < 60:
            score = 70
            score += 10 if last["Close"] > last["EMA50"] else 0
            score += 10 if last["RSI"] > prev["RSI"] else 0  # RSI turning up
            score += 5 if last["ATR_pct"] < df["ATR_pct"].rolling(50).mean().iloc[-1] else 0
            candidates.append({
                "side": "BUY", "entry": last["Close"],
                "stop": last["Close"] - 1.5 * last["ATR"],
                "target": last["Close"] + 2.5 * last["ATR"],
                "score": min(score, 100), "strategy": "TrendPullback",
                "atr": last["ATR"], "rsi": last["RSI"], **meta
            })

    if trend_down:
        distance = (last["EMA20"] - last["Close"]) / last["ATR"]
        if -0.5 < distance < 0.5 and 40 < last["RSI"] < 60:
            score = 70
            score += 10 if last["Close"] < last["EMA50"] else 0
            score += 10 if last["RSI"] < prev["RSI"] else 0
            score += 5 if last["ATR_pct"] < df["ATR_pct"].rolling(50).mean().iloc[-1] else 0
            candidates.append({
                "side": "SELL", "entry": last["Close"],
                "stop": last["Close"] + 1.5 * last["ATR"],
                "target": last["Close"] - 2.5 * last["ATR"],
                "score": min(score, 100), "strategy": "TrendPullback",
                "atr": last["ATR"], "rsi": last["RSI"], **meta
            })

    # --- STRATEGY 2: Strong momentum breakout (secondary) ---
    recent_high = df["High"].iloc[-20:-1].max()
    recent_low  = df["Low"].iloc[-20:-1].min()
    body = abs(last["Close"] - last["Open"])
    avg_body = (df["Close"] - df["Open"]).abs().rolling(20).mean().iloc[-1]

    if last["Close"] > recent_high and body > 1.5 * avg_body and last["RSI"] < 75:
        score = 60
        score += 15 if trend_up else 0
        score += 10 if last["Close"] > last["EMA20"] else 0
        candidates.append({
            "side": "BUY", "entry": last["Close"],
            "stop": last["Close"] - 1.5 * last["ATR"],
            "target": last["Close"] + 3.0 * last["ATR"],
            "score": min(score, 100), "strategy": "Breakout",
            "atr": last["ATR"], "rsi": last["RSI"], **meta
        })

    if last["Close"] < recent_low and body > 1.5 * avg_body and last["RSI"] > 25:
        score = 60
        score += 15 if trend_down else 0
        score += 10 if last["Close"] < last["EMA20"] else 0
        candidates.append({
            "side": "SELL", "entry": last["Close"],
            "stop": last["Close"] + 1.5 * last["ATR"],
            "target": last["Close"] - 3.0 * last["ATR"],
            "score": min(score, 100), "strategy": "Breakout",
            "atr": last["ATR"], "rsi": last["RSI"], **meta
        })

    return candidates


def main():
    state = load_state()
    reset_daily_if_new_day(state)
    save_state(state)

    risk = get_risk_status(state)

    print("=" * 70)
    print(f"  MULTI-ASSET SCANNER  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print(f"  Equity: ${risk['equity']:.2f}  |  Daily: ${risk['daily_pnl']:+.2f}  |  Total: ${risk['total_pnl']:+.2f}")
    print(f"  Remaining daily risk:  ${risk['remaining_daily_risk']:.2f}")
    print(f"  Remaining total risk:  ${risk['remaining_total_risk']:.2f}")
    print(f"  Trading days: {risk['trading_days']}/5 minimum")
    print("=" * 70)

    if risk["halted"]:
        print("🛑 HALTED — do not trade. Rule limit reached.")
        return
    if risk["target_reached"]:
        print("🎯 TARGET REACHED — challenge passed.")
        return

    print(f"\nScanning {len(INSTRUMENTS)} instruments...\n")

    all_signals = []
    for ticker, meta in INSTRUMENTS.items():
        try:
            df = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 200:
                continue
            df = add_indicators(df)
            signals = score_signal(df, meta)
            all_signals.extend(signals)
        except Exception as e:
            print(f"  {meta['name']}: skip ({e})")

    if not all_signals:
        print("  No qualifying setups across all instruments.")
        print("  ✅ This is fine. Patience is the edge. Check again in 1 hour.")
        return

    # Sort by score descending, take top 3
    all_signals.sort(key=lambda x: x["score"], reverse=True)
    top = all_signals[:3]

    print(f"  Found {len(all_signals)} candidate(s). Top {len(top)}:\n")

    for i, sig in enumerate(top, 1):
        risk_dollars = min(
            risk["equity"] * 0.01,
            risk["remaining_daily_risk"] * 0.5,
            risk["remaining_total_risk"] * 0.5,
        )
        stop_distance = abs(sig["entry"] - sig["stop"])
        stop_pips = stop_distance / sig["pip"]
        lots = risk_dollars / (stop_pips * sig["pip_value"]) if stop_pips > 0 else 0
        pips_target = abs(sig["target"] - sig["entry"]) / sig["pip"]
        rr = pips_target / stop_pips if stop_pips > 0 else 0

        print("  " + "─" * 66)
        print(f"  #{i}  [{sig['score']}/100]  {sig['side']} {sig['name']}  — {sig['strategy']}")
        print("  " + "─" * 66)
        print(f"    Entry:    {sig['entry']:.5f}")
        print(f"    Stop:     {sig['stop']:.5f}  ({stop_pips:.1f} pips, ${risk_dollars:.2f} risk)")
        print(f"    Target:   {sig['target']:.5f}  ({pips_target:.1f} pips, R:R 1:{rr:.1f})")
        print(f"    Lots:     {round(max(0.01, lots), 2)}  (verify on platform!)")
        print(f"    RSI:      {sig['rsi']:.1f}")
        print()

    print("=" * 70)
    print("  RULE: Take ONE trade. If it wins, stop for the day.")
    print("        If it loses, you may take ONE more. Then stop.")
    print("=" * 70)


if __name__ == "__main__":
    main()