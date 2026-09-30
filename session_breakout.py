import pandas as pd
import numpy as np

PAIRS = {
    "GBPUSD=X": {"name": "GBPUSD", "pip": 0.0001, "pip_value": 10.0, "spread": 1.2},
    "EURUSD=X": {"name": "EURUSD", "pip": 0.0001, "pip_value": 10.0, "spread": 1.0},
    "USDJPY=X": {"name": "USDJPY", "pip": 0.01,   "pip_value": 6.7,  "spread": 1.0},
}


def add_atr(df, period=14):
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


def get_uk_hour_utc(ts):
    """Approximate UK hour. UTC+0 winter, UTC+1 summer. We'll use UTC+1 for BST."""
    # Simple approximation: May-Oct = BST (UTC+1), else GMT (UTC+0)
    month = ts.month
    offset = 1 if 4 <= month <= 10 else 0
    return (ts.hour + offset) % 24


def signal_london_breakout(df, i):
    """
    London Open breakout:
    - Asian range = 00:00-07:00 UK
    - Entry window: 07:00-11:00 UK (any bar)
    - Breakout: close breaks Asian high/low
    - Stop: opposite side of range
    - Target: 1x range width
    """
    if i < 20:
        return None

    ts = df.index[i]
    hour_uk = get_uk_hour_utc(ts)

    # Only fire during London window 07:00-10:59 UK
    if hour_uk < 7 or hour_uk > 10:
        return None

    today = ts.date()

    # Get Asian session bars (00:00-06:59 UK) for today
    asian_bars = []
    for k in range(max(0, i - 12), i):
        bar_ts = df.index[k]
        if bar_ts.date() != today:
            continue
        bar_hour = get_uk_hour_utc(bar_ts)
        if 0 <= bar_hour < 7:
            asian_bars.append(k)

    if len(asian_bars) < 3:
        return None

    range_high = df["High"].iloc[asian_bars].max()
    range_low  = df["Low"].iloc[asian_bars].min()
    range_width = range_high - range_low

    atr = df["ATR"].iloc[i]
    if pd.isna(atr) or atr <= 0:
        return None

    # Range sanity: not dead, not wild
    if range_width < 0.10 * atr or range_width > 1.5 * atr:
        return None

    close = df["Close"].iloc[i]

    if close > range_high:
        return {
            "side": "BUY",
            "entry": close,
            "stop": range_low,
            "target": close + range_width,
            "range_width": range_width,
            "hour_uk": hour_uk,
        }
    if close < range_low:
        return {
            "side": "SELL",
            "entry": close,
            "stop": range_high,
            "target": close - range_width,
            "range_width": range_width,
            "hour_uk": hour_uk,
        }
    return None


def simulate(df, i, sig, meta, risk_dollars):
    """Enter at next bar's open. Walk until SL or TP. Max hold 20 bars."""
    if i + 1 >= len(df):
        return None

    entry = df["Open"].iloc[i + 1]
    spread = meta["spread"] * meta["pip"]

    if sig["side"] == "BUY":
        entry += spread
        stop_dist = sig["entry"] - sig["stop"]
        target_dist = sig["target"] - sig["entry"]
        stop = entry - stop_dist
        target = entry + target_dist
        for j in range(i + 2, min(i + 22, len(df))):
            if df["Low"].iloc[j] <= stop:
                return j, -risk_dollars, "loss"
            if df["High"].iloc[j] >= target:
                rr = target_dist / stop_dist if stop_dist > 0 else 1
                return j, risk_dollars * rr, "win"
        # Timeout close
        if i + 21 < len(df):
            exit_price = df["Close"].iloc[i + 21]
            pips = (exit_price - entry) / meta["pip"]
            stop_pips = stop_dist / meta["pip"]
            pnl = risk_dollars * (pips / stop_pips) if stop_pips > 0 else 0
            return i + 21, pnl, "timeout"
    else:
        entry -= spread
        stop_dist = sig["stop"] - sig["entry"]
        target_dist = sig["entry"] - sig["target"]
        stop = entry + stop_dist
        target = entry - target_dist
        for j in range(i + 2, min(i + 22, len(df))):
            if df["High"].iloc[j] >= stop:
                return j, -risk_dollars, "loss"
            if df["Low"].iloc[j] <= target:
                rr = target_dist / stop_dist if stop_dist > 0 else 1
                return j, risk_dollars * rr, "win"
        if i + 21 < len(df):
            exit_price = df["Close"].iloc[i + 21]
            pips = (entry - exit_price) / meta["pip"]
            stop_pips = stop_dist / meta["pip"]
            pnl = risk_dollars * (pips / stop_pips) if stop_pips > 0 else 0
            return i + 21, pnl, "timeout"

    return None