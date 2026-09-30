import yfinance as yf
import pandas as pd
from strategies import INSTRUMENTS
from strategies_v2 import add_indicators, signal_mean_reversion_loose

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
PAIRS = ["GBPUSD", "EURUSD"]


def simulate(df, i, sig, meta, risk_dollars):
    if i + 1 >= len(df):
        return None
    entry = df["Open"].iloc[i + 1]
    spread = meta["spread"] * meta["pip"]
    min_hold = sig.get("min_hold", 1)

    if sig["side"] == "BUY":
        entry += spread
        stop = sig["stop"] + (entry - df["Close"].iloc[i])
        target = sig["target"] + (entry - df["Close"].iloc[i])
        for j in range(i + 2, len(df)):
            if j < i + 1 + min_hold:
                continue
            if df["Low"].iloc[j] <= stop:
                return j, -risk_dollars, "loss"
            if df["High"].iloc[j] >= target:
                rp = abs(target - entry) / meta["pip"]
                sp = abs(entry - stop) / meta["pip"]
                return j, risk_dollars * rp / sp, "win"
    else:
        entry -= spread
        stop = sig["stop"] - (df["Close"].iloc[i] - entry)
        target = sig["target"] - (df["Close"].iloc[i] - entry)
        for j in range(i + 2, len(df)):
            if j < i + 1 + min_hold:
                continue
            if df["High"].iloc[j] >= stop:
                return j, -risk_dollars, "loss"
            if df["Low"].iloc[j] <= target:
                rp = abs(entry - target) / meta["pip"]
                sp = abs(stop - entry) / meta["pip"]
                return j, risk_dollars * rp / sp, "win"
    return None


def backtest(start, end, risk_pct):
    all_trades = []
    for ticker, meta in INSTRUMENTS.items():
        if meta["name"] not in PAIRS:
            continue
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 250:
            continue
        df = add_indicators(df)
        i = 200
        while i < len(df) - 10:
            sig = signal_mean_reversion_loose(df, i)
            if sig is None:
                i += 1
                continue
            r = simulate(df, i, sig, meta, ACCOUNT_SIZE * risk_pct)
            if r is None:
                i += 1
                continue
            exit_idx, pnl, outcome = r
            all_trades.append({
                "symbol": meta["name"], "pnl": pnl,
                "entry_time": df.index[i + 1], "exit_time": df.index[exit_idx],
            })
            i = exit_idx + 2
    return all_trades


def print_raw_stats(trades):
    if not trades:
        print("    No trades")
        return
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 999
    wr = len(wins) / len(trades) * 100
    print(f"    Total trades:  {len(trades)}")
    print(f"    Win rate:      {wr:.1f}%")
    print(f"    Profit factor: {pf:.2f}")
    print(f"    Total P&L:     ${sum(pnls):+.2f}")


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
    first_entry = None

    for t in trades:
        d = t["exit_time"].date()
        if current_day != d:
            current_day = d
            daily_start = equity
            if reason == "daily":
                halted = False
                reason = None
        if halted:
            continue
        projected = equity + t["pnl"]
        if projected - daily_start < -DAILY_LOSS_LIMIT:
            continue
        if first_entry is None:
            first_entry = t["entry_time"]
        equity = projected
        hwm = max(hwm, equity)
        if not trailing_locked and equity >= ACCOUNT_SIZE + MAX_DD:
            trailing_locked = True
        accepted.append({**t, "equity_after": equity})
        floor = ACCOUNT_SIZE if trailing_locked else max(hwm - MAX_DD, ACCOUNT_SIZE - MAX_DD)
        if equity <= floor:
            halted = True
            reason = "max_dd"
        if equity - daily_start <= -DAILY_LOSS_LIMIT:
            halted = True
            reason = "daily"
        if equity >= ACCOUNT_SIZE + TARGET:
            halted = True
            reason = "target"

    days = (accepted[-1]["exit_time"] - first_entry).days if accepted and first_entry is not None else 0
    return accepted, equity, halted, reason, days


def print_challenge(label, trades):
    accepted, eq, halted, reason, days = apply_rules(trades)
    print(f"\n  == {label} ==")
    if not accepted:
        print("    No trades survived.")
        return
    pnls = [t["pnl"] for t in accepted]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 999
    print(f"    Trades: {len(accepted)}   Win%: {len(wins)/len(accepted)*100:.1f}   PF: {pf:.2f}")
    print(f"    Final: ${eq:.2f}   Halted: {halted} ({reason})")
    if reason == "target":
        print(f"    PASSED in {days} days ({days/30:.1f} months)")
    elif reason == "max_dd":
        print(f"    FAILED - Max DD")


def main():
    print("=" * 78)
    print("  LOOSENED FILTER - More trades, more robustness")
    print("=" * 78)

    periods = [
        ("2015-2018", "2015-01-01", "2018-01-01"),
        ("2018-2021", "2018-01-01", "2021-01-01"),
        ("2021-2024", "2021-01-01", "2024-01-01"),
        ("2023-2026", "2023-01-01", "2026-01-01"),
    ]

    print("\n### RAW strategy stats (no challenge rules) — 2% risk equivalent")
    for label, s, e in periods:
        trades = backtest(s, e, 0.02)
        print(f"\n  == {label} raw ==")
        print_raw_stats(trades)

    print("\n\n### CHALLENGE SIMULATION — 2% risk")
    for label, s, e in periods:
        trades = backtest(s, e, 0.02)
        print_challenge(label, trades)

    print("\n\n### FULL 2015-2026 combined")
    trades = backtest("2015-01-01", "2026-01-01", 0.02)
    print(f"\n  Raw stats:")
    print_raw_stats(trades)
    print_challenge("Full period", trades)


if __name__ == "__main__":
    main()
