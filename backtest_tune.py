import yfinance as yf
import pandas as pd
import os
import time

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0
TICKER = "EURUSD=X"
META = {"name": "EURUSD", "pip": 0.0001, "spread": 1.0}
ALLOWED_HOURS = {9}
CACHE_FILE = "eurusd_1h_cache.csv"


def add_atr(df, period=14):
    df = df.copy()
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


def load_data():
    """Load from cache if exists, otherwise download and cache."""
    if os.path.exists(CACHE_FILE):
        print(f"  Using cached data: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return add_atr(df)

    print("  Downloading from Yahoo (may hit rate limits)...")
    for attempt in range(3):
        try:
            df = yf.download(TICKER, period="730d", interval="1h",
                             progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                df.to_csv(CACHE_FILE)
                print(f"  Cached {len(df)} bars to {CACHE_FILE}")
                return add_atr(df)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
        time.sleep(15)
    print("  All download attempts failed. Wait a few minutes and retry.")
    return None


def get_uk_hour(ts):
    offset = 1 if 4 <= ts.month <= 10 else 0
    return (ts.hour + offset) % 24


def detect(df, i):
    if i < 20: return None
    ts = df.index[i]
    if get_uk_hour(ts) not in ALLOWED_HOURS: return None
    today = ts.date()
    idx = [k for k in range(max(0, i-12), i)
           if df.index[k].date() == today and 0 <= get_uk_hour(df.index[k]) < 7]
    if len(idx) < 3: return None
    rh = df["High"].iloc[idx].max()
    rl = df["Low"].iloc[idx].min()
    rw = rh - rl
    atr = df["ATR"].iloc[i]
    if pd.isna(atr) or atr <= 0: return None
    if rw < 0.10 * atr or rw > 1.5 * atr: return None
    c = df["Close"].iloc[i]
    if c > rh: return {"side": "BUY", "range_high": rh, "range_low": rl, "range_width": rw}
    if c < rl: return {"side": "SELL", "range_high": rh, "range_low": rl, "range_width": rw}
    return None


def simulate(df, i, sig, risk, stop_mode):
    if i + 1 >= len(df): return None
    entry = df["Open"].iloc[i + 1]
    spread = META["spread"] * META["pip"]

    if sig["side"] == "BUY":
        entry += spread
        if stop_mode == "range":
            stop = sig["range_low"]
        elif stop_mode == "tight":
            stop = sig["range_high"]
        elif stop_mode == "mid":
            stop = (sig["range_high"] + sig["range_low"]) / 2
        else:
            stop = sig["range_low"]
        target = entry + sig["range_width"]
        sd = entry - stop
        if sd <= 0: return None
        td = target - entry
        for j in range(i+2, min(i+22, len(df))):
            if df["Low"].iloc[j] <= stop: return j, -risk, "loss"
            if df["High"].iloc[j] >= target:
                return j, risk * (td / sd), "win"
        if i+21 < len(df):
            ep = df["Close"].iloc[i+21]
            return i+21, risk * ((ep-entry)/META["pip"]) / (sd/META["pip"]), "timeout"
    else:
        entry -= spread
        if stop_mode == "range":
            stop = sig["range_high"]
        elif stop_mode == "tight":
            stop = sig["range_low"]
        elif stop_mode == "mid":
            stop = (sig["range_high"] + sig["range_low"]) / 2
        else:
            stop = sig["range_high"]
        target = entry - sig["range_width"]
        sd = stop - entry
        if sd <= 0: return None
        td = entry - target
        for j in range(i+2, min(i+22, len(df))):
            if df["High"].iloc[j] >= stop: return j, -risk, "loss"
            if df["Low"].iloc[j] <= target:
                return j, risk * (td / sd), "win"
        if i+21 < len(df):
            ep = df["Close"].iloc[i+21]
            return i+21, risk * ((entry-ep)/META["pip"]) / (sd/META["pip"]), "timeout"
    return None


def backtest(df, risk, stop_mode):
    trades = []
    last_day = None
    i = 20
    while i < len(df) - 25:
        sig = detect(df, i)
        if sig is None: i += 1; continue
        entry_day = df.index[i+1].date()
        if last_day == entry_day: i += 1; continue
        r = simulate(df, i, sig, risk, stop_mode)
        if r is None: i += 1; continue
        exit_idx, pnl, outcome = r
        trades.append({"pnl": pnl, "outcome": outcome,
                       "entry_time": df.index[i+1], "exit_time": df.index[exit_idx]})
        last_day = entry_day
        i = exit_idx + 2
    return trades


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


def report(label, trades):
    if not trades: print(f"  {label:<28} no trades"); return
    pnls = [t["pnl"] for t in trades]
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p < 0]
    pf = (sum(w)/abs(sum(l))) if l else 999
    print(f"  {label:<28} n={len(trades):>4}  win%={len(w)/len(trades)*100:>5.1f}  "
          f"PF={pf:>5.2f}  AvgW=${sum(w)/len(w) if w else 0:>5.2f}  "
          f"AvgL=${sum(l)/len(l) if l else 0:>6.2f}  P&L=${sum(pnls):>+7.2f}")


def challenge(label, trades):
    acc, eq, h, r, days = apply_rules(trades)
    if not acc: print(f"  {label:<28} none survived"); return
    pnls = [t["pnl"] for t in acc]
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p < 0]
    pf = (sum(w)/abs(sum(l))) if l else 999
    st = "PASSED" if r == "target" else f"FAILED({r})"
    print(f"  {label:<28} n={len(acc):>3}  win%={len(w)/len(acc)*100:>5.1f}  PF={pf:>5.2f}  "
          f"final=${eq:>6.2f}  days={days:>3}  {st}")


def main():
    print("=" * 100)
    print("  TUNING — EURUSD + 09:00 UK")
    print("=" * 100)
    df = load_data()
    if df is None:
        return
    print(f"  Data: {len(df)} bars\n")

    print("  --- RAW STATS ---")
    for sm in ["range", "mid", "tight"]:
        for risk in [5.0, 7.5, 10.0]:
            report(f"stop={sm} risk=${risk}", backtest(df, risk, sm))

    print("\n  --- CHALLENGE SIM ---")
    for sm in ["range", "mid", "tight"]:
        for risk in [5.0, 7.5, 10.0]:
            challenge(f"stop={sm} risk=${risk}", backtest(df, risk, sm))


if __name__ == "__main__":
    main()