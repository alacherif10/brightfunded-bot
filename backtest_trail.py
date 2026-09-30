import yfinance as yf
import pandas as pd
import os

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
RISK = 10.0

META = {"name": "EURUSD", "pip": 0.0001, "spread": 1.0}
ALLOWED_HOURS = {9, 10}
CACHE = "EURUSD_1h_cache.csv"


def add_atr(df, period=14):
    df = df.copy()
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    # Also add an ATR for trailing (faster)
    df["ATR_trail"] = tr.rolling(7).mean()
    return df


def get_uk_hour(ts):
    offset = 1 if 4 <= ts.month <= 10 else 0
    return (ts.hour + offset) % 24


def load_data():
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df[df.index.notna()]
        df.index = df.index.tz_localize(None)
    else:
        print(f"  Downloading EURUSD...")
        df = yf.download("EURUSD=X", period="730d", interval="1h",
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(CACHE)
    return add_atr(df)


def detect(df, i):
    if i < 20: return None
    ts = df.index[i]
    if get_uk_hour(ts) not in ALLOWED_HOURS: return None
    today = ts.date()
    idx = [k for k in range(max(0, i - 12), i)
           if df.index[k].date() == today and 0 <= get_uk_hour(df.index[k]) < 7]
    if len(idx) < 3: return None
    rh = df["High"].iloc[idx].max()
    rl = df["Low"].iloc[idx].min()
    rw = rh - rl
    atr = df["ATR"].iloc[i]
    if pd.isna(atr) or atr <= 0: return None
    if rw < 0.10 * atr or rw > 1.5 * atr: return None
    c = df["Close"].iloc[i]
    if c > rh:
        return {"side": "BUY", "range_high": rh, "range_low": rl, "range_width": rw}
    if c < rl:
        return {"side": "SELL", "range_high": rh, "range_low": rl, "range_width": rw}
    return None


def simulate_trail(df, i, sig, meta, mode):
    """
    Enter at next open. Trail stop as price moves.
    mode: 'atr'  -> trail by 1×ATR_trail
          'range'-> trail by half range_width
          'be_then_atr' -> move to BE at +1R, then trail by ATR
    Max hold = 30 bars (~30 hours, covering NY session + next Asian)
    """
    if i + 1 >= len(df): return None

    entry = df["Open"].iloc[i + 1]
    spread = meta["spread"] * meta["pip"]
    rw = sig["range_width"]

    if sig["side"] == "BUY":
        entry += spread
        initial_stop = sig["range_low"]
        initial_risk = entry - initial_stop
        if initial_risk <= 0: return None
        stop = initial_stop
        max_favorable = entry  # track for logging

        for j in range(i + 2, min(i + 32, len(df))):
            bar_high = df["High"].iloc[j]
            bar_low = df["Low"].iloc[j]

            # Check stop first (conservative)
            if bar_low <= stop:
                pnl_pips = (stop - entry) / meta["pip"]
                pnl_dollars = RISK * (stop - entry) / initial_risk
                return j, pnl_dollars, "stopped"

            max_favorable = max(max_favorable, bar_high)

            # Trail the stop
            if mode == "atr":
                new_stop = bar_high - 1.0 * df["ATR_trail"].iloc[j]
            elif mode == "range":
                new_stop = bar_high - 0.5 * rw
            elif mode == "be_then_atr":
                # Move to BE once we're +1R, then trail by ATR
                if bar_high >= entry + initial_risk:
                    new_stop = max(entry, bar_high - 1.0 * df["ATR_trail"].iloc[j])
                else:
                    new_stop = stop
            else:
                new_stop = stop

            # Stop only moves up for BUY
            if new_stop > stop:
                stop = new_stop

        # Timeout close
        if i + 31 < len(df):
            ep = df["Close"].iloc[i + 31]
            pnl_dollars = RISK * (ep - entry) / initial_risk
            return i + 31, pnl_dollars, "timeout"
    else:
        entry -= spread
        initial_stop = sig["range_high"]
        initial_risk = initial_stop - entry
        if initial_risk <= 0: return None
        stop = initial_stop

        for j in range(i + 2, min(i + 32, len(df))):
            bar_high = df["High"].iloc[j]
            bar_low = df["Low"].iloc[j]

            if bar_high >= stop:
                pnl_dollars = RISK * (entry - stop) / initial_risk
                return j, pnl_dollars, "stopped"

            if mode == "atr":
                new_stop = bar_low + 1.0 * df["ATR_trail"].iloc[j]
            elif mode == "range":
                new_stop = bar_low + 0.5 * rw
            elif mode == "be_then_atr":
                if bar_low <= entry - initial_risk:
                    new_stop = min(entry, bar_low + 1.0 * df["ATR_trail"].iloc[j])
                else:
                    new_stop = stop
            else:
                new_stop = stop

            if new_stop < stop:
                stop = new_stop

        if i + 31 < len(df):
            ep = df["Close"].iloc[i + 31]
            pnl_dollars = RISK * (entry - ep) / initial_risk
            return i + 31, pnl_dollars, "timeout"
    return None


def backtest(df, mode):
    trades = []
    last_day = None
    i = 20
    while i < len(df) - 35:
        sig = detect(df, i)
        if sig is None: i += 1; continue
        entry_day = df.index[i + 1].date()
        if last_day == entry_day: i += 1; continue
        r = simulate_trail(df, i, sig, META, mode)
        if r is None: i += 1; continue
        exit_idx, pnl, outcome = r
        trades.append({
            "pnl": pnl, "outcome": outcome,
            "entry_time": df.index[i + 1], "exit_time": df.index[exit_idx],
        })
        last_day = entry_day
        i = exit_idx + 2
    return trades


def report(label, trades):
    if not trades: print(f"  {label:<22} no trades"); return
    pnls = [t["pnl"] for t in trades]
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p < 0]
    pf = (sum(w) / abs(sum(l))) if l else 999
    biggest = max(pnls)
    print(f"  {label:<22} n={len(trades):>4}  win%={len(w)/len(trades)*100:>5.1f}  "
          f"PF={pf:>5.2f}  AvgW=${sum(w)/len(w) if w else 0:>5.2f}  "
          f"AvgL=${sum(l)/len(l) if l else 0:>6.2f}  MaxW=${biggest:>+7.2f}  "
          f"P&L=${sum(pnls):>+8.2f}")


def apply_rules(trades):
    trades = sorted(trades, key=lambda t: t["exit_time"])
    equity = ACCOUNT_SIZE; daily_start = ACCOUNT_SIZE; hwm = ACCOUNT_SIZE
    trailing_locked = False; current_day = None; halted = False; reason = None
    accepted = []; days = set()
    for t in trades:
        d = t["exit_time"].date()
        if current_day != d:
            current_day = d; daily_start = equity
            if reason == "daily": halted = False; reason = None
        if halted: continue
        proj = equity + t["pnl"]
        if proj - daily_start < -DAILY_LOSS_LIMIT: continue
        equity = proj; hwm = max(hwm, equity)
        if not trailing_locked and equity >= ACCOUNT_SIZE + MAX_DD: trailing_locked = True
        days.add(d); accepted.append({**t, "equity_after": equity})
        floor = ACCOUNT_SIZE if trailing_locked else max(hwm - MAX_DD, ACCOUNT_SIZE - MAX_DD)
        if equity <= floor: halted = True; reason = "max_dd"
        if equity - daily_start <= -DAILY_LOSS_LIMIT: halted = True; reason = "daily"
        if equity >= ACCOUNT_SIZE + TARGET: halted = True; reason = "target"
    return accepted, equity, halted, reason, len(days)


def challenge(label, trades):
    acc, eq, h, r, days = apply_rules(trades)
    if not acc: print(f"  {label:<22} none survived"); return
    pnls = [t["pnl"] for t in acc]
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p < 0]
    pf = (sum(w) / abs(sum(l))) if l else 999
    st = "PASSED" if r == "target" else f"FAILED({r})"
    print(f"  {label:<22} n={len(acc):>3}  win%={len(w)/len(acc)*100:>5.1f}  PF={pf:>5.2f}  "
          f"final=${eq:>6.2f}  days={days:>3}  {st}")


def main():
    print("=" * 100)
    print("  EURUSD 09:00-10:59 UK — TRAILING STOP TEST")
    print("=" * 100)
    df = load_data()
    print(f"  Data: {len(df)} bars\n")

    print("  --- RAW STATS (3 trail modes) ---")
    for mode in ["atr", "range", "be_then_atr"]:
        trades = backtest(df, mode)
        report(f"trail={mode}", trades)

    print("\n  --- CHALLENGE SIM ---")
    for mode in ["atr", "range", "be_then_atr"]:
        trades = backtest(df, mode)
        challenge(f"trail={mode}", trades)

    print("\n  --- WINNERS DISTRIBUTION (trail=atr) ---")
    trades = backtest(df, "atr")
    wins = sorted([t["pnl"] for t in trades if t["pnl"] > 0], reverse=True)
    if wins:
        print(f"    Top 5 winners: {[f'${w:.2f}' for w in wins[:5]]}")
        print(f"    Total from top 5: ${sum(wins[:5]):.2f}")
        print(f"    Total from all winners: ${sum(wins):.2f}")


if __name__ == "__main__":
    main()
