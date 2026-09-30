import yfinance as yf
import pandas as pd
import os

ACCOUNT_SIZE = 1000.0
DAILY_LOSS_LIMIT = 30.0
MAX_DD = 60.0
TARGET = 100.0

PAIRS = {
    "EURUSD=X": {"name": "EURUSD", "pip": 0.0001, "spread": 1.0},
    "GBPUSD=X": {"name": "GBPUSD", "pip": 0.0001, "spread": 1.2},
    "USDCHF=X": {"name": "USDCHF", "pip": 0.0001, "spread": 1.2},
    "EURJPY=X": {"name": "EURJPY", "pip": 0.01,   "spread": 1.5},
}

# Test matrix
HOUR_SETS = [(9, 10), (9, 10, 11)]
RISK_LEVELS = [10.0, 15.0, 20.0]


def add_atr(df, period=14):
    df = df.copy()
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


def get_uk_hour(ts):
    offset = 1 if 4 <= ts.month <= 10 else 0
    return (ts.hour + offset) % 24


def load_data():
    data = {}
    for ticker, meta in PAIRS.items():
        cache = f"{meta['name']}_1h_cache.csv"
        if os.path.exists(cache):
            df = pd.read_csv(cache, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
            df = df[df.index.notna()]
            df.index = df.index.tz_localize(None)
        else:
            print(f"  Downloading {meta['name']}...")
            df = yf.download(ticker, period="730d", interval="1h",
                             progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(cache)
        data[ticker] = (add_atr(df), meta)
    return data


def detect(df, i, allowed_hours):
    if i < 20: return None
    ts = df.index[i]
    if get_uk_hour(ts) not in allowed_hours: return None
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
    if c > rh: return {"side": "BUY", "range_high": rh, "range_low": rl, "range_width": rw}
    if c < rl: return {"side": "SELL", "range_high": rh, "range_low": rl, "range_width": rw}
    return None


def simulate_range_trail(df, i, sig, meta, risk):
    """Trail by half the Asian range."""
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

        for j in range(i + 2, min(i + 32, len(df))):
            if df["Low"].iloc[j] <= stop:
                return j, risk * (stop - entry) / initial_risk, "stopped"
            new_stop = df["High"].iloc[j] - 0.5 * rw
            if new_stop > stop: stop = new_stop

        if i + 31 < len(df):
            ep = df["Close"].iloc[i + 31]
            return i + 31, risk * (ep - entry) / initial_risk, "timeout"
    else:
        entry -= spread
        initial_stop = sig["range_high"]
        initial_risk = initial_stop - entry
        if initial_risk <= 0: return None
        stop = initial_stop

        for j in range(i + 2, min(i + 32, len(df))):
            if df["High"].iloc[j] >= stop:
                return j, risk * (entry - stop) / initial_risk, "stopped"
            new_stop = df["Low"].iloc[j] + 0.5 * rw
            if new_stop < stop: stop = new_stop

        if i + 31 < len(df):
            ep = df["Close"].iloc[i + 31]
            return i + 31, risk * (entry - ep) / initial_risk, "timeout"
    return None


def backtest(data, allowed_hours, risk, pairs_filter=None):
    all_trades = []
    for ticker, (df, meta) in data.items():
        if pairs_filter and meta["name"] not in pairs_filter:
            continue
        last_day = None
        i = 20
        while i < len(df) - 35:
            sig = detect(df, i, allowed_hours)
            if sig is None: i += 1; continue
            entry_day = df.index[i + 1].date()
            if last_day == entry_day: i += 1; continue
            r = simulate_range_trail(df, i, sig, meta, risk)
            if r is None: i += 1; continue
            exit_idx, pnl, outcome = r
            all_trades.append({
                "symbol": meta["name"], "pnl": pnl,
                "entry_time": df.index[i + 1], "exit_time": df.index[exit_idx],
            })
            last_day = entry_day
            i = exit_idx + 2
    return all_trades


def report(label, trades):
    if not trades: print(f"  {label:<32} no trades"); return
    pnls = [t["pnl"] for t in trades]
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p < 0]
    pf = (sum(w) / abs(sum(l))) if l else 999
    print(f"  {label:<32} n={len(trades):>4}  win%={len(w)/len(trades)*100:>5.1f}  "
          f"PF={pf:>5.2f}  P&L=${sum(pnls):>+8.2f}")


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
    if not acc: print(f"  {label:<32} none survived"); return
    pnls = [t["pnl"] for t in acc]
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p < 0]
    pf = (sum(w) / abs(sum(l))) if l else 999
    st = "PASSED ✅" if r == "target" else f"FAILED({r})"
    print(f"  {label:<32} n={len(acc):>3}  win%={len(w)/len(acc)*100:>5.1f}  PF={pf:>5.2f}  "
          f"final=${eq:>6.2f}  days={days:>3}  {st}")


def main():
    print("=" * 100)
    print("  SCALING TEST — trail=range on 4 pairs, 3 risk levels, 2 hour windows")
    print("=" * 100)
    data = load_data()
    print()

    # --- Test 1: EURUSD only (baseline, verify consistency) ---
    print("  --- A) EURUSD only (baseline) ---")
    for risk in RISK_LEVELS:
        t = backtest(data, (9, 10), risk, pairs_filter=["EURUSD"])
        report(f"EURUSD only, risk=${risk}", t)

    print("\n  --- B) All 4 pairs (diversification) ---")
    for risk in RISK_LEVELS:
        t = backtest(data, (9, 10), risk)
        report(f"4 pairs, risk=${risk}", t)

    print("\n  --- C) All 4 pairs + 11am entry ---")
    for risk in RISK_LEVELS:
        t = backtest(data, (9, 10, 11), risk)
        report(f"4 pairs+11am, risk=${risk}", t)

    print("\n" + "=" * 100)
    print("  CHALLENGE SIM — Best configurations")
    print("=" * 100)

    print("\n  --- EURUSD only, risk=$15 ---")
    t = backtest(data, (9, 10), 15.0, pairs_filter=["EURUSD"])
    challenge("EURUSD risk=$15", t)

    print("\n  --- EURUSD only, risk=$20 ---")
    t = backtest(data, (9, 10), 20.0, pairs_filter=["EURUSD"])
    challenge("EURUSD risk=$20", t)

    print("\n  --- 4 pairs, risk=$15 ---")
    t = backtest(data, (9, 10), 15.0)
    challenge("4 pairs risk=$15", t)

    print("\n  --- 4 pairs, risk=$20 ---")
    t = backtest(data, (9, 10), 20.0)
    challenge("4 pairs risk=$20", t)

    print("\n  --- 4 pairs + 11am, risk=$15 ---")
    t = backtest(data, (9, 10, 11), 15.0)
    challenge("4p+11am risk=$15", t)


if __name__ == "__main__":
    main()
