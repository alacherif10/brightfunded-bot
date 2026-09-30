import yfinance as yf
import pandas as pd
from collections import defaultdict
from session_breakout import PAIRS, add_atr, signal_london_breakout, simulate

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
RISK_PER_TRADE = 10.0


def fetch_data():
    data = {}
    for ticker, meta in PAIRS.items():
        print(f"  Fetching {meta['name']}...")
        df = yf.download(ticker, period="730d", interval="1h",
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 100:
            print("    not enough data")
            continue
        df = add_atr(df)
        data[ticker] = (df, meta)
        print(f"    {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return data


def backtest_slice(df, meta, start, end):
    """Run on one instrument within a date slice."""
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
        entry_day = df_work.index[i + 1].date()
        # Only one trade per day per instrument
        if last_entry_day == entry_day:
            i += 1
            continue
        r = simulate(df_work, i, sig, meta, RISK_PER_TRADE)
        if r is None:
            i += 1
            continue
        exit_idx, pnl, outcome = r
        trades.append({
            "symbol": meta["name"], "pnl": pnl,
            "entry_time": df_work.index[i + 1],
            "exit_time": df_work.index[exit_idx],
            "outcome": outcome,
            "hour_uk": sig["hour_uk"],
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
    print("  LONDON OPEN BREAKOUT — Last 730 days (Yahoo 1h limit)")
    print("=" * 78)

    data = fetch_data()
    if not data:
        return

    all_dates = pd.concat([df.index.to_series() for df, _ in data.values()])
    global_start = all_dates.min()
    global_end = all_dates.max()
    print(f"\n  Data range: {global_start.date()} -> {global_end.date()}")

    # Quarterly slices
    total_days = (global_end - global_start).days
    step = total_days // 4
    slices = []
    for q in range(4):
        s = global_start + pd.Timedelta(days=q * step)
        e = global_start + pd.Timedelta(days=(q + 1) * step) if q < 3 else global_end
        slices.append((f"Q{q+1} {s.date()}", s, e))

    print("\n  --- Quarterly (raw edge) ---")
    all_trades = []
    for label, s, e in slices:
        period_trades = []
        for ticker, (df, meta) in data.items():
            period_trades.extend(backtest_slice(df, meta, s, e))
        raw_stats(label, period_trades)
        all_trades.extend(period_trades)

    print("\n  --- Full period ---")
    raw_stats("ALL", all_trades)

    print("\n  --- Per instrument ---")
    by_sym = defaultdict(list)
    for t in all_trades:
        by_sym[t["symbol"]].append(t["pnl"])
    for sym, sp in sorted(by_sym.items(), key=lambda x: -sum(x[1])):
        w = sum(1 for p in sp if p > 0)
        win_sum = sum(p for p in sp if p > 0)
        loss_sum = abs(sum(p for p in sp if p < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 999
        print(f"    {sym:<8} {len(sp):>4} trades   win%: {w/len(sp)*100:>5.1f}   "
              f"PF: {pf:>5.2f}   P&L: ${sum(sp):>+8.2f}")

    print("\n  --- By entry hour (UK time) ---")
    by_hour = defaultdict(list)
    for t in all_trades:
        by_hour[t["hour_uk"]].append(t["pnl"])
    for hour in sorted(by_hour.keys()):
        sp = by_hour[hour]
        w = sum(1 for p in sp if p > 0)
        print(f"    {hour:02d}:00 UK   {len(sp):>4} trades   "
              f"win%: {w/len(sp)*100:>5.1f}   P&L: ${sum(sp):>+8.2f}")

    print("\n" + "=" * 78)
    print("  CHALLENGE SIMULATION (BrightFunded $1k)")
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


if __name__ == "__main__":
    main()