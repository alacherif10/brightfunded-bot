import yfinance as yf
import pandas as pd
from collections import defaultdict
from strategies import INSTRUMENTS, add_indicators, signal_mean_reversion

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0


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


def backtest(tickers, start, end, risk_pct):
    """Run backtest between two dates with given risk % per trade."""
    all_trades = []
    for ticker, meta in INSTRUMENTS.items():
        if meta["name"] not in tickers:
            continue
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 250:
                continue
            df = add_indicators(df)

            i = 200
            while i < len(df) - 10:
                sig = signal_mean_reversion(df, i)
                if sig is None:
                    i += 1
                    continue
                risk_dollars = ACCOUNT_SIZE * risk_pct
                r = simulate(df, i, sig, meta, risk_dollars)
                if r is None:
                    i += 1
                    continue
                exit_idx, pnl, outcome = r
                all_trades.append({
                    "symbol": meta["name"], "pnl": pnl,
                    "entry_time": df.index[i + 1], "exit_time": df.index[exit_idx],
                })
                i = exit_idx + 2
        except Exception as e:
            print(f"  {meta['name']}: {e}")
    return all_trades


def apply_rules(trades, risk_pct):
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
    first_entry = None

    for t in trades:
        d = t["exit_time"].date() if hasattr(t["exit_time"], "date") else t["exit_time"]
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
        if first_entry is None:
            first_entry = t["entry_time"]
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

    return accepted, equity, halted, reason, len(days), first_entry


def print_result(label, accepted, final_eq, halted, reason, days, first_entry):
    print(f"\n  ── {label} ──")
    if not accepted:
        print("    No trades.")
        return
    pnls = [t["pnl"] for t in accepted]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 999
    print(f"    Trades: {len(accepted)}   Win%: {len(wins)/len(accepted)*100:.1f}   PF: {pf:.2f}")
    print(f"    Final equity: ${final_eq:.2f}   Halted: {halted} ({reason})")
    if first_entry is not None and accepted:
        last = accepted[-1]["exit_time"]
        span = (last - first_entry).days
        print(f"    Days to finish: {span}")
        if reason == "target":
            print(f"    ✅ PASSED in {span} days")


def main():
    print("=" * 78)
    print("  WALK-FORWARD VALIDATION + RISK SIZING TEST")
    print("=" * 78)

    # ---- Test 1: Walk-forward ----
    print("\n### TEST 1: Walk-forward (all 5 pairs, 1% risk)")
    print("\n  In-sample: 2019-2021")
    is_trades = backtest(["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"],
                         "2019-01-01", "2021-01-01", 0.01)
    acc, eq, h, r, d, fe = apply_rules(is_trades, 0.01)
    print_result("In-sample 2019-2021", acc, eq, h, r, d, fe)

    print("\n  Out-of-sample: 2022-2024")
    oos_trades = backtest(["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"],
                          "2022-01-01", "2024-12-31", 0.01)
    acc, eq, h, r, d, fe = apply_rules(oos_trades, 0.01)
    print_result("Out-of-sample 2022-2024", acc, eq, h, r, d, fe)

    # ---- Test 2: Risk sizing ----
    print("\n### TEST 2: Risk sizing (2020-2025, all pairs)")
    for risk in [0.01, 0.02, 0.03]:
        print(f"\n  --- Risk per trade: {risk*100:.0f}% (${ACCOUNT_SIZE * risk:.2f}) ---")
        trades = backtest(["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"],
                          "2020-01-01", "2025-01-01", risk)
        acc, eq, h, r, d, fe = apply_rules(trades, risk)
        print_result(f"Risk {risk*100:.0f}%", acc, eq, h, r, d, fe)

    # ---- Test 3: GBPUSD only (best performer) ----
    print("\n### TEST 3: GBPUSD + EURUSD only (best edge pairs)")
    for risk in [0.01, 0.02]:
        trades = backtest(["GBPUSD", "EURUSD"], "2020-01-01", "2025-01-01", risk)
        acc, eq, h, r, d, fe = apply_rules(trades, risk)
        print_result(f"GBPUSD+EURUSD, risk {risk*100:.0f}%", acc, eq, h, r, d, fe)


if __name__ == "__main__":
    main()