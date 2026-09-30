import yfinance as yf
import pandas as pd
from collections import defaultdict
from strategies import (
    INSTRUMENTS, add_indicators,
    signal_mean_reversion, signal_session_breakout, signal_h4_trend,
)

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
RISK_PER_TRADE = 10.0

STRATEGIES = {
    "MeanRev": signal_mean_reversion,
    "Session": signal_session_breakout,
    "Trend":   signal_h4_trend,
}


def simulate(df, i, sig, meta):
    """Enter next bar open, walk forward until SL/TP."""
    if i + 1 >= len(df):
        return None
    entry = df["Open"].iloc[i + 1]
    spread = meta["spread"] * meta["pip"]

    if sig["side"] == "BUY":
        entry += spread
        stop = entry - (sig["entry_offset_stop"] if "entry_offset_stop" in sig else abs(df["Close"].iloc[i] - sig["stop"]))
        stop = sig["stop"] + (entry - df["Close"].iloc[i])
        target = sig["target"] + (entry - df["Close"].iloc[i])
        min_hold = sig.get("min_hold", 1)

        for j in range(i + 2, len(df)):
            if j < i + 1 + min_hold:
                continue
            if df["Low"].iloc[j] <= stop:
                pips = abs(entry - stop) / meta["pip"]
                return j, -RISK_PER_TRADE, pips, "loss"
            if df["High"].iloc[j] >= target:
                reward_pips = abs(target - entry) / meta["pip"]
                stop_pips = abs(entry - stop) / meta["pip"]
                return j, RISK_PER_TRADE * reward_pips / stop_pips, reward_pips, "win"
    else:
        entry -= spread
        stop = sig["stop"] - (df["Close"].iloc[i] - entry)
        target = sig["target"] - (df["Close"].iloc[i] - entry)
        min_hold = sig.get("min_hold", 1)

        for j in range(i + 2, len(df)):
            if j < i + 1 + min_hold:
                continue
            if df["High"].iloc[j] >= stop:
                return j, -RISK_PER_TRADE, 0, "loss"
            if df["Low"].iloc[j] <= target:
                reward_pips = abs(entry - target) / meta["pip"]
                stop_pips = abs(stop - entry) / meta["pip"]
                return j, RISK_PER_TRADE * reward_pips / stop_pips, reward_pips, "win"
    return None


def backtest_strategy(name, sig_fn, interval, period_days):
    print(f"\n  === Strategy: {name} ({interval} bars, {period_days} days) ===")
    all_trades = []

    for ticker, meta in INSTRUMENTS.items():
        try:
            df = yf.download(ticker, period=f"{period_days}d", interval=interval,
                             progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 250:
                continue
            df = add_indicators(df)

            i = 200
            while i < len(df) - 10:
                sig = sig_fn(df, i)
                if sig is None:
                    i += 1
                    continue
                r = simulate(df, i, sig, meta)
                if r is None:
                    i += 1
                    continue
                exit_idx, pnl, pips, outcome = r
                all_trades.append({
                    "symbol": meta["name"], "strategy": name,
                    "entry_time": df.index[i + 1], "exit_time": df.index[exit_idx],
                    "pnl": pnl, "outcome": outcome,
                })
                i = exit_idx + 2
        except Exception as e:
            print(f"    {meta['name']}: {e}")

    return all_trades


def print_stats(name, trades):
    if not trades:
        print(f"  {name}: no trades")
        return
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 999
    print(f"  {name}: {len(trades):>4} trades   "
          f"win%: {len(wins)/len(trades)*100:>5.1f}   "
          f"PF: {pf:>5.2f}   P&L: ${sum(pnls):>+8.2f}")


def main():
    print("=" * 78)
    print("  STRATEGY COMPARISON — 3 Different Approaches")
    print("=" * 78)

    results = {}

    # Mean reversion on H4
    trades = backtest_strategy("MeanRev-H4", signal_mean_reversion, "4h", 730)
    results["MeanRev-H4"] = trades

    # Session breakout on H1
    trades = backtest_strategy("Session-H1", signal_session_breakout, "1h", 730)
    results["Session-H1"] = trades

    # H4 trend following
    trades = backtest_strategy("Trend-H4", signal_h4_trend, "4h", 730)
    results["Trend-H4"] = trades

    # Daily mean reversion (coarser)
    trades = backtest_strategy("MeanRev-D", signal_mean_reversion, "1d", 2000)
    results["MeanRev-D"] = trades

    print("\n" + "=" * 78)
    print("  FINAL COMPARISON")
    print("=" * 78)
    for name, trades in results.items():
        print_stats(name, trades)

    print("\n  Which one has the best edge? Look for:")
    print("    - PF > 1.3 (meaningful edge)")
    print("    - Win rate high enough for its R:R")
    print("    - Enough trades to be statistically meaningful (>50)")


if __name__ == "__main__":
    main()