import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# ============ CONFIG ============
ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
RISK_PER_TRADE = 10.0
SPREAD_PIPS = {
    "EURUSD": 1.0, "GBPUSD": 1.2, "USDJPY": 1.0, "USDCHF": 1.2,
    "USDCAD": 1.5, "AUDUSD": 1.2, "NZDUSD": 1.5, "EURGBP": 1.5,
    "EURJPY": 1.5, "GBPJPY": 2.0, "XAUUSD": 3.0, "XAGUSD": 3.0,
}
INSTRUMENTS = {
    "EURUSD=X":  {"name": "EURUSD", "pip": 0.0001, "pip_value": 10.0},
    "GBPUSD=X":  {"name": "GBPUSD", "pip": 0.0001, "pip_value": 10.0},
    "USDJPY=X":  {"name": "USDJPY", "pip": 0.01,   "pip_value": 6.7},
    "USDCHF=X":  {"name": "USDCHF", "pip": 0.0001, "pip_value": 11.0},
    "USDCAD=X":  {"name": "USDCAD", "pip": 0.0001, "pip_value": 7.3},
    "AUDUSD=X":  {"name": "AUDUSD", "pip": 0.0001, "pip_value": 10.0},
    "EURGBP=X":  {"name": "EURGBP", "pip": 0.0001, "pip_value": 13.0},
    "EURJPY=X":  {"name": "EURJPY", "pip": 0.01,   "pip_value": 6.7},
    "GBPJPY=X":  {"name": "GBPJPY", "pip": 0.01,   "pip_value": 6.7},
    "GC=F":      {"name": "XAUUSD", "pip": 0.10,   "pip_value": 10.0},
}
# ================================


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
    return df


def detect_signal(df, i):
    """Detect signal at bar i using ONLY data up to bar i."""
    if i < 200:
        return None
    last = df.iloc[i]
    prev = df.iloc[i - 1]

    atr = last["ATR"]
    if pd.isna(atr) or atr <= 0:
        return None

    trend_up   = last["EMA50"] > last["EMA200"] and last["EMA20"] > last["EMA50"]
    trend_down = last["EMA50"] < last["EMA200"] and last["EMA20"] < last["EMA50"]

    # TrendPullback
    if trend_up:
        distance = (last["Close"] - last["EMA20"]) / atr
        if -0.3 < distance < 0.3 and 45 < last["RSI"] < 55 and last["RSI"] > prev["RSI"]:
            return {"side": "BUY", "strategy": "TrendPullback", "atr": atr}
    if trend_down:
        distance = (last["EMA20"] - last["Close"]) / atr
        if -0.3 < distance < 0.3 and 45 < last["RSI"] < 55 and last["RSI"] < prev["RSI"]:
            return {"side": "SELL", "strategy": "TrendPullback", "atr": atr}

    # Breakout
    recent_high = df["High"].iloc[i-20:i].max()
    recent_low  = df["Low"].iloc[i-20:i].min()
    body = abs(last["Close"] - last["Open"])
    avg_body = (df["Close"] - df["Open"]).abs().rolling(20).mean().iloc[i]

    if pd.isna(avg_body) or avg_body <= 0:
        return None

    if last["Close"] > recent_high and body > 2.0 * avg_body and last["RSI"] < 70:
        return {"side": "BUY", "strategy": "Breakout", "atr": atr}
    if last["Close"] < recent_low and body > 2.0 * avg_body and last["RSI"] > 30:
        return {"side": "SELL", "strategy": "Breakout", "atr": atr}

    return None


def simulate_trade(df, entry_idx, signal, meta, risk_dollars):
    """Enter at next bar's open. Simulate until SL or TP."""
    if entry_idx + 1 >= len(df):
        return None

    entry = df["Open"].iloc[entry_idx + 1]
    spread = SPREAD_PIPS.get(meta["name"], 1.0) * meta["pip"]
    atr = signal["atr"]

    if signal["side"] == "BUY":
        entry += spread
        stop = entry - 1.5 * atr
        target = entry + 2.5 * atr
        for j in range(entry_idx + 2, len(df)):
            low = df["Low"].iloc[j]
            high = df["High"].iloc[j]
            if low <= stop:
                return j, stop, -risk_dollars, "loss"
            if high >= target:
                return j, target, risk_dollars * 2.5 / 1.5, "win"
    else:
        entry -= spread
        stop = entry + 1.5 * atr
        target = entry - 2.5 * atr
        for j in range(entry_idx + 2, len(df)):
            low = df["Low"].iloc[j]
            high = df["High"].iloc[j]
            if high >= stop:
                return j, stop, -risk_dollars, "loss"
            if low <= target:
                return j, target, risk_dollars * 2.5 / 1.5, "win"

    return None


def backtest_symbol(ticker, meta):
    """Run backtest on one instrument using while loop to avoid overlapping trades."""
    print(f"  Downloading {meta['name']}...")
    df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 250:
        print(f"    Not enough data for {meta['name']}")
        return []

    df = add_indicators(df)
    trades = []

    i = 200
    while i < len(df) - 5:
        sig = detect_signal(df, i)
        if sig is None:
            i += 1
            continue

        result = simulate_trade(df, i, sig, meta, RISK_PER_TRADE)
        if result is None:
            i += 1
            continue

        exit_idx, exit_price, pnl, outcome = result
        trades.append({
            "symbol": meta["name"],
            "strategy": sig["strategy"],
            "side": sig["side"],
            "entry_time": df.index[i + 1],
            "exit_time": df.index[exit_idx],
            "pnl": pnl,
            "outcome": outcome,
        })
        i = exit_idx + 2

    return trades


def apply_brightfunded_rules(trades):
    """Simulate trades through BrightFunded challenge rules."""
    trades = sorted(trades, key=lambda t: t["exit_time"])

    equity = ACCOUNT_SIZE
    daily_start = ACCOUNT_SIZE
    hwm = ACCOUNT_SIZE
    trailing_locked = False
    current_day = None
    halted = False
    halt_reason = None
    accepted = []
    days_traded = set()

    for t in trades:
        exit_day = t["exit_time"].date() if hasattr(t["exit_time"], "date") else t["exit_time"]

        if current_day != exit_day:
            current_day = exit_day
            daily_start = equity
            if halt_reason == "daily":
                halted = False
                halt_reason = None

        if halted:
            continue

        projected = equity + t["pnl"]
        if projected - daily_start < -DAILY_LOSS_LIMIT:
            continue

        equity = projected
        hwm = max(hwm, equity)
        if not trailing_locked and equity >= ACCOUNT_SIZE + MAX_DD:
            trailing_locked = True

        days_traded.add(exit_day)
        accepted.append({**t, "equity_after": equity})

        floor = ACCOUNT_SIZE if trailing_locked else max(hwm - MAX_DD, ACCOUNT_SIZE - MAX_DD)
        if equity <= floor:
            halted = True
            halt_reason = "max_dd"
        if equity - daily_start <= -DAILY_LOSS_LIMIT:
            halted = True
            halt_reason = "daily"
        if equity >= ACCOUNT_SIZE + TARGET:
            halted = True
            halt_reason = "target"

    return accepted, {
        "final_equity": equity,
        "halted": halted,
        "halt_reason": halt_reason,
        "days_traded": len(days_traded),
    }


def print_raw_stats(trades):
    """RAW stats — no challenge filter. Measures actual strategy edge."""
    if not trades:
        print("\nNo trades generated.")
        return
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    print("\n" + "=" * 70)
    print("  RAW STRATEGY STATS (no challenge filter - measures real edge)")
    print("=" * 70)
    print(f"  Total trades:      {len(trades)}")
    print(f"  Win rate:          {len(wins) / len(trades) * 100:.1f}%")
    print(f"  Total P&L:         ${sum(pnls):+.2f}")
    if wins:
        print(f"  Avg win:           ${sum(wins)/len(wins):+.2f}")
    if losses:
        print(f"  Avg loss:          ${sum(losses)/len(losses):+.2f}")
        pf = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else 999
        print(f"  Profit factor:     {pf:.2f}")

    print("\n  Per instrument (raw):")
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t["pnl"])
    for sym, sp in sorted(by_sym.items(), key=lambda x: -sum(x[1])):
        w = sum(1 for p in sp if p > 0)
        win_sum = sum(p for p in sp if p > 0)
        loss_sum = abs(sum(p for p in sp if p < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 999
        print(f"    {sym:<8}  {len(sp):>4} trades   ${sum(sp):>+8.2f}   win%: {w/len(sp)*100:>3.0f}   PF: {pf:.2f}")

    print("\n  Per strategy (raw):")
    by_strat = defaultdict(list)
    for t in trades:
        by_strat[t["strategy"]].append(t["pnl"])
    for strat, sp in sorted(by_strat.items()):
        w = sum(1 for p in sp if p > 0)
        win_sum = sum(p for p in sp if p > 0)
        loss_sum = abs(sum(p for p in sp if p < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 999
        print(f"    {strat:<16}  {len(sp):>4} trades   ${sum(sp):>+8.2f}   win%: {w/len(sp)*100:>3.0f}   PF: {pf:.2f}")
    print("=" * 70)


def print_summary(all_trades, result):
    if not all_trades:
        print("\nNo trades survived BrightFunded rules.")
        return

    pnls = [t["pnl"] for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    print("\n" + "=" * 70)
    print("  BRIGHTFUNDED CHALLENGE SIMULATION")
    print("=" * 70)
    print(f"  Trades taken:      {len(all_trades)}")
    print(f"  Wins:              {len(wins)}")
    print(f"  Losses:            {len(losses)}")
    print(f"  Win rate:          {len(wins) / len(all_trades) * 100:.1f}%")
    print(f"  Total P&L:         ${sum(pnls):+.2f}")
    print(f"  Final equity:      ${result['final_equity']:.2f}")
    print(f"  Halted:            {result['halted']} ({result['halt_reason']})")
    print(f"  Days with trades:  {result['days_traded']}")

    if wins:
        print(f"  Avg win:           ${sum(wins)/len(wins):+.2f}")
    if losses:
        print(f"  Avg loss:          ${sum(losses)/len(losses):+.2f}")
        pf = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else 999
        print(f"  Profit factor:     {pf:.2f}")

    print("\n" + "=" * 70)
    print("  VERDICT:")
    if result["halt_reason"] == "target":
        print("  PASSED the challenge in backtest.")
    elif result["halt_reason"] == "max_dd":
        print("  Breached max drawdown. Strategy does not work on $1k account.")
    elif result["halt_reason"] == "daily":
        print("  Hit daily loss limit. Too aggressive for $1k rules.")
    else:
        print("  Did not reach target. Needs more time or better edge.")
    print("=" * 70)


def main():
    print("Running backtest across all instruments...\n")
    all_trades = []
    for ticker, meta in INSTRUMENTS.items():
        try:
            trades = backtest_symbol(ticker, meta)
            all_trades.extend(trades)
            print(f"    -> {len(trades)} trades")
        except Exception as e:
            print(f"    Error on {meta['name']}: {e}")

    # 1. Raw edge (no rules) — does the strategy make money?
    print_raw_stats(all_trades)

    # 2. Challenge simulation — would it pass the $1k BrightFunded rules?
    accepted, result = apply_brightfunded_rules(all_trades)
    print_summary(accepted, result)


if __name__ == "__main__":
    main()