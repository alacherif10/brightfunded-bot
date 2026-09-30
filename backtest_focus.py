import yfinance as yf
import pandas as pd
from collections import defaultdict
from strategies import INSTRUMENTS, add_indicators, signal_mean_reversion

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
RISK_PER_TRADE = 10.0


def simulate(df, i, sig, meta):
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
                return j, -RISK_PER_TRADE, "loss"
            if df["High"].iloc[j] >= target:
                rp = abs(target - entry) / meta["pip"]
                sp = abs(entry - stop) / meta["pip"]
                return j, RISK_PER_TRADE * rp / sp, "win"
    else:
        entry -= spread
        stop = sig["stop"] - (df["Close"].iloc[i] - entry)
        target = sig["target"] - (df["Close"].iloc[i] - entry)
        for j in range(i + 2, len(df)):
            if j < i + 1 + min_hold:
                continue
            if df["High"].iloc[j] >= stop:
                return j, -RISK_PER_TRADE, "loss"
            if df["Low"].iloc[j] <= target:
                rp = abs(entry - target) / meta["pip"]
                sp = abs(stop - entry) / meta["pip"]
                return j, RISK_PER_TRADE * rp / sp, "win"
    return None


def run(period_days=2000):
    print(f"Running MeanRev-D on {period_days} days of daily data...\n")
    all_trades = []
    per_symbol = defaultdict(list)

    for ticker, meta in INSTRUMENTS.items():
        try:
            df = yf.download(ticker, period=f"{period_days}d", interval="1d",
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
                r = simulate(df, i, sig, meta)
                if r is None:
                    i += 1
                    continue
                exit_idx, pnl, outcome = r
                trade = {
                    "symbol": meta["name"], "side": sig["side"],
                    "entry_time": df.index[i + 1], "exit_time": df.index[exit_idx],
                    "pnl": pnl, "outcome": outcome,
                }
                all_trades.append(trade)
                per_symbol[meta["name"]].append(trade)
                i = exit_idx + 2
        except Exception as e:
            print(f"  {meta['name']}: {e}")

    return all_trades, per_symbol


def print_per_symbol(per_symbol):
    print("=" * 78)
    print("  PER-SYMBOL BREAKDOWN — MeanRev-D")
    print("=" * 78)
    print(f"  {'Symbol':<10} {'Trades':>7} {'Win%':>7} {'PF':>6} {'P&L':>10} {'AvgWin':>9} {'AvgLoss':>9}")
    print("  " + "-" * 74)
    for sym in sorted(per_symbol.keys()):
        trades = per_symbol[sym]
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else 999
        aw = (sum(wins) / len(wins)) if wins else 0
        al = (sum(losses) / len(losses)) if losses else 0
        print(f"  {sym:<10} {len(trades):>7} {len(wins)/len(trades)*100:>6.1f}% "
              f"{pf:>6.2f} ${sum(pnls):>+9.2f} ${aw:>+8.2f} ${al:>+8.2f}")


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
        d = t["exit_time"].date() if hasattr(t["exit_time"], "date") else t["exit_time"]
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


def print_challenge(accepted, final_eq, halted, reason, days):
    print("\n" + "=" * 78)
    print("  BRIGHTFUNDED $1K CHALLENGE SIMULATION — MeanRev-D")
    print("=" * 78)
    if not accepted:
        print("  No trades survived.")
        return
    pnls = [t["pnl"] for t in accepted]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 999
    print(f"  Trades taken:     {len(accepted)}")
    print(f"  Win rate:         {len(wins)/len(accepted)*100:.1f}%")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Final equity:     ${final_eq:.2f}  (started $1,000.00)")
    print(f"  Halted:           {halted}  ({reason})")
    print(f"  Days with trades: {days}")
    if accepted:
        first = accepted[0]["exit_time"]
        last = accepted[-1]["exit_time"]
        if hasattr(first, "date"):
            days_span = (last - first).days
            print(f"  Period covered:   {first.date()}  →  {last.date()}  ({days_span} days)")
    print("=" * 78)
    if reason == "target":
        print("  ✅ PASSES the challenge in backtest.")
    elif reason == "max_dd":
        print("  ❌ Still breaches max drawdown.")
    elif reason == "daily":
        print("  ⚠️  Hits daily loss limit.")
    else:
        print("  ⏳ Doesn't reach target in the test period.")


if __name__ == "__main__":
    trades, per_symbol = run(2000)
    print_per_symbol(per_symbol)
    accepted, final_eq, halted, reason, days = apply_rules(trades)
    print_challenge(accepted, final_eq, halted, reason, days)
