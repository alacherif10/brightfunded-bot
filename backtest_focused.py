import yfinance as yf
import pandas as pd
from session_breakout import add_atr, signal_london_breakout, simulate
from collections import defaultdict

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
RISK_PER_TRADE = 10.0

# Focused config based on what the data told us
TICKER = "EURUSD=X"
META = {"name": "EURUSD", "pip": 0.0001, "pip_value": 10.0, "spread": 1.0}
ALLOWED_HOURS = {9}   # only 09:00 UK entries


def fetch():
    df = yf.download(TICKER, period="730d", interval="1h",
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return add_atr(df)


def backtest(df, start, end):
    df_slice = df.loc[(df.index >= start) & (df.index < end)]
    if len(df_slice) < 40:
        return []
    start_loc = df.index.get_loc(df_slice.index[0])
    warmup = max(0, start_loc - 30)
    df_work = df.iloc[warmup:]

    trades = []
    last_entry_day = None
    i = 20
    while i < len(df_work) - 25:
        sig = signal_london_breakout(df_work, i)
        if sig is None:
            i += 1
            continue
        # Focused filter: only allowed hours
        if sig["hour_uk"] not in ALLOWED_HOURS:
            i += 1
            continue
        entry_day = df_work.index[i + 1].date()
        if last_entry_day == entry_day:
            i += 1
            continue
        r = simulate(df_work, i, sig, META, RISK_PER_TRADE)
        if r is None:
            i += 1
            continue
        exit_idx, pnl, outcome = r
        trades.append({
            "symbol": META["name"], "pnl": pnl,
            "entry_time": df_work.index[i + 1],
            "exit_time": df_work.index[exit_idx],
            "outcome": outcome,
        })
        last_entry_day = entry_day
        i = exit_idx + 2
    return trades


def raw_stats(label, trades):
    if not trades:
        print(f"  {label:<20} no trades")
        return
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 999
    wr = len(wins) / len(trades) * 100
    print(f"  {label:<20} trades: {len(trades):>4}   win%: {wr:>5.1f}   "
          f"PF: {pf:>5.2f}   P&L: ${sum(pnls):>+8.2f}")


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
                halted = False
                reason = None
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


def main():
    print("=" * 78)
    print("  FOCUSED STRATEGY — EURUSD + 09:00 UK window")
    print("=" * 78)

    df = fetch()
    print(f"  Data: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")

    global_start = df.index[0]
    global_end = df.index[-1]
    total_days = (global_end - global_start).days
    step = total_days // 4
    slices = []
    for q in range(4):
        s = global_start + pd.Timedelta(days=q * step)
        e = global_start + pd.Timedelta(days=(q + 1) * step) if q < 3 else global_end
        slices.append((f"Q{q+1} {s.date()}", s, e))

    print("\n  --- Quarterly breakdown ---")
    all_trades = []
    for label, s, e in slices:
        trades = backtest(df, s, e)
        raw_stats(label, trades)
        all_trades.extend(trades)

    print("\n  --- Full period ---")
    raw_stats("ALL", all_trades)

    print("\n" + "=" * 78)
    print("  CHALLENGE SIMULATION")
    print("=" * 78)
    accepted, eq, halted, reason, days = apply_rules(all_trades)
    if accepted:
        pnls = [t["pnl"] for t in accepted]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else 999
        print(f"  Trades: {len(accepted)}   Win%: {len(wins)/len(accepted)*100:.1f}   PF: {pf:.2f}")
        print(f"  Final: ${eq:.2f}   Halted: {halted} ({reason})   Days: {days}")
        if reason == "target":
            print("  PASSED")
        elif reason == "max_dd":
            print("  FAILED - Max DD")
    else:
        print("  No trades survived rules")


if __name__ == "__main__":
    main()
