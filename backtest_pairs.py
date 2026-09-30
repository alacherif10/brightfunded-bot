import yfinance as yf
import pandas as pd
import numpy as np
from pairs_strategy import PAIRS, add_indicators, compute_spread_zscore, signal_pairs

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
RISK_PER_TRADE = 10.0   # $10 per leg


def simulate_pair(df_a, df_b, i, sig, meta_a, meta_b, risk):
    """
    Enter both legs next bar. Exit when Z-score crosses back through 0
    OR when hard stop hit on combined position.
    """
    if i + 2 >= len(df_a) or i + 2 >= len(df_b):
        return None

    entry_a = df_a["Open"].iloc[i + 1]
    entry_b = df_b["Open"].iloc[i + 1]

    # Hard stops: 2 ATR each leg
    if sig["side_a"] == "BUY":
        entry_a += meta_a["spread"] * meta_a["pip"]
        stop_a = entry_a - 2.0 * sig["atr_a"]
    else:
        entry_a -= meta_a["spread"] * meta_a["pip"]
        stop_a = entry_a + 2.0 * sig["atr_a"]

    if sig["side_b"] == "BUY":
        entry_b += meta_b["spread"] * meta_b["pip"]
        stop_b = entry_b - 2.0 * sig["atr_b"]
    else:
        entry_b -= meta_b["spread"] * meta_b["pip"]
        stop_b = entry_b + 2.0 * sig["atr_b"]

    # Precompute the Z-score series for the whole df (this is fine — it's rolling)
    _, z_series = compute_spread_zscore(df_a, df_b)

    for j in range(i + 2, len(df_a) - 1):
        # Check hard stops
        if sig["side_a"] == "BUY":
            if df_a["Low"].iloc[j] <= stop_a:
                return j, -risk, "stop_a"
        else:
            if df_a["High"].iloc[j] >= stop_a:
                return j, -risk, "stop_a"

        if sig["side_b"] == "BUY":
            if df_b["Low"].iloc[j] <= stop_b:
                return j, -risk, "stop_b"
        else:
            if df_b["High"].iloc[j] >= stop_b:
                return j, -risk, "stop_b"

        # Check exit condition: Z crossed 0
        z_val = z_series.iloc[j] if j < len(z_series) else None
        if z_val is not None and not pd.isna(z_val):
            if sig["z"] > 0 and z_val <= 0:
                # Profitable reversal to mean
                pnl = risk * 1.2   # approximate — assumes 1.2R avg on z-reversion
                return j, pnl, "mean_reversion"
            if sig["z"] < 0 and z_val >= 0:
                pnl = risk * 1.2
                return j, pnl, "mean_reversion"

    return None


def backtest_pair(pair, start, end):
    ticker_a, ticker_b = pair
    name_a = ticker_a.replace("=X", "")
    name_b = ticker_b.replace("=X", "")

    df_a = yf.download(ticker_a, start=start, end=end, interval="1d",
                       progress=False, auto_adjust=False)
    df_b = yf.download(ticker_b, start=start, end=end, interval="1d",
                       progress=False, auto_adjust=False)
    if isinstance(df_a.columns, pd.MultiIndex):
        df_a.columns = df_a.columns.get_level_values(0)
    if isinstance(df_b.columns, pd.MultiIndex):
        df_b.columns = df_b.columns.get_level_values(0)

    if df_a.empty or df_b.empty or len(df_a) < 100:
        return []

    df_a = add_indicators(df_a)
    df_b = add_indicators(df_b)

    meta_a = {"name": name_a, "pip": 0.0001, "spread": 1.0}
    meta_b = {"name": name_b, "pip": 0.0001, "spread": 1.2}

    trades = []
    i = 70
    while i < len(df_a) - 5:
        sig = signal_pairs(df_a, df_b, i)
        if sig is None:
            i += 1
            continue
        r = simulate_pair(df_a, df_b, i, sig, meta_a, meta_b, RISK_PER_TRADE)
        if r is None:
            i += 1
            continue
        exit_idx, pnl, outcome = r
        trades.append({
            "pair": f"{name_a}/{name_b}",
            "entry_time": df_a.index[i + 1],
            "exit_time": df_a.index[exit_idx],
            "pnl": pnl,
            "outcome": outcome,
            "z_entry": sig["z"],
        })
        i = exit_idx + 2

    return trades


def apply_rules(trades):
    trades = sorted(trades, key=lambda t: t["exit_time"])
    equity = ACCOUNT_SIZE
    daily_start = ACCOUNT_SIZE
    hwm = ACCOUNT_SIZE
    trailing_locked = False
    current_day = None
    halted = False
    reason = None
    accepted = []
    days = set()

    for t in trades:
        d = t["exit_time"].date()
        if current_day != d:
            current_day = d
            daily_start = equity
            if reason == "daily":
                halted = False; reason = None
        if halted:
            continue
        projected = equity + t["pnl"]
        if projected - daily_start < -DAILY_LOSS_LIMIT:
            continue
        equity = projected
        hwm = max(hwm, equity)
        if not trailing_locked and equity >= ACCOUNT_SIZE + MAX_DD:
            trailing_locked = True
        days.add(d)
        accepted.append({**t, "equity_after": equity})
        floor = ACCOUNT_SIZE if trailing_locked else max(hwm - MAX_DD, ACCOUNT_SIZE - MAX_DD)
        if equity <= floor:
            halted = True; reason = "max_dd"
        if equity - daily_start <= -DAILY_LOSS_LIMIT:
            halted = True; reason = "daily"
        if equity >= ACCOUNT_SIZE + TARGET:
            halted = True; reason = "target"

    return accepted, equity, halted, reason, len(days)


def print_raw(label, trades):
    if not trades:
        print(f"\n  == {label} ==  no trades")
        return
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 999
    print(f"\n  == {label} ==")
    print(f"    Trades: {len(trades)}   Win%: {len(wins)/len(trades)*100:.1f}   "
          f"PF: {pf:.2f}   P&L: ${sum(pnls):+.2f}")


def main():
    print("=" * 78)
    print("  PAIRS TRADING BACKTEST (statistical arbitrage)")
    print("=" * 78)

    periods = [
        ("2015-2018", "2015-01-01", "2018-01-01"),
        ("2018-2021", "2018-01-01", "2021-01-01"),
        ("2021-2024", "2021-01-01", "2024-01-01"),
        ("2023-2026", "2023-01-01", "2026-01-01"),
    ]

    all_trades_combined = []
    for label, s, e in periods:
        period_trades = []
        for pair in PAIRS:
            try:
                trades = backtest_pair(pair, s, e)
                period_trades.extend(trades)
            except Exception as ex:
                print(f"    error on {pair}: {ex}")
        print_raw(label, period_trades)
        all_trades_combined.extend(period_trades)

    print("\n" + "=" * 78)
    print("  CHALLENGE SIMULATION (all periods combined)")
    print("=" * 78)
    accepted, eq, halted, reason, days = apply_rules(all_trades_combined)
    if accepted:
        pnls = [t["pnl"] for t in accepted]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else 999
        print(f"  Trades: {len(accepted)}   Win%: {len(wins)/len(accepted)*100:.1f}   PF: {pf:.2f}")
        print(f"  Final: ${eq:.2f}   Halted: {halted} ({reason})   Days traded: {days}")
        if reason == "target":
            print("  ✅ PASSED")
        elif reason == "max_dd":
            print("  ❌ Max DD breach")
    print("=" * 78)


if __name__ == "__main__":
    main()
