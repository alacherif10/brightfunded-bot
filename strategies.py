import yfinance as yf
import pandas as pd
import numpy as np

# ============ INSTRUMENTS ============
# Focus on pairs that mean-revert most reliably
INSTRUMENTS = {
    "EURUSD=X": {"name": "EURUSD", "pip": 0.0001, "pip_value": 10.0, "spread": 1.0},
    "GBPUSD=X": {"name": "GBPUSD", "pip": 0.0001, "pip_value": 10.0, "spread": 1.2},
    "USDJPY=X": {"name": "USDJPY", "pip": 0.01,   "pip_value": 6.7,  "spread": 1.0},
    "AUDUSD=X": {"name": "AUDUSD", "pip": 0.0001, "pip_value": 10.0, "spread": 1.2},
    "USDCAD=X": {"name": "USDCAD", "pip": 0.0001, "pip_value": 7.3,  "spread": 1.5},
}


def add_indicators(df):
    df = df.copy()
    df["EMA20"]  = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]  = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Distance from mean in ATR units
    df["dist_from_mean"] = (df["Close"] - df["EMA20"]) / df["ATR"]
    return df


# ============ STRATEGY 1: Daily Mean-Reversion ============
def signal_mean_reversion(df, i):
    """
    Buy when price is >2 ATR BELOW EMA20 (oversold panic) in a range regime.
    Sell when price is >2 ATR ABOVE EMA20 (overbought euphoria) in a range regime.
    Target: return to EMA20.
    """
    if i < 200:
        return None
    last = df.iloc[i]
    if pd.isna(last["ATR"]) or last["ATR"] <= 0:
        return None

    # Regime filter: not in strong trend (EMA50 close to EMA200)
    ema_spread = abs(last["EMA50"] - last["EMA200"]) / last["ATR"]
    if ema_spread > 3.0:   # trending — skip mean reversion
        return None

    dist = last["dist_from_mean"]

    # Oversold → BUY
    if dist < -2.0 and last["RSI"] < 30:
        target = last["EMA20"]
        stop_dist = 1.0 * last["ATR"]
        return {
            "side": "BUY", "strategy": "MeanRev",
            "atr": last["ATR"],
            "stop": last["Close"] - stop_dist,
            "target": target,
            "min_hold": 4,   # hold at least 4 bars
        }

    # Overbought → SELL
    if dist > 2.0 and last["RSI"] > 70:
        target = last["EMA20"]
        stop_dist = 1.0 * last["ATR"]
        return {
            "side": "SELL", "strategy": "MeanRev",
            "atr": last["ATR"],
            "stop": last["Close"] + stop_dist,
            "target": target,
            "min_hold": 4,
        }
    return None


# ============ STRATEGY 2: Session Breakout ============
def signal_session_breakout(df, i):
    """
    Breakout of the previous 12-hour high/low during London (7-10 UK)
    or New York (13-16 UK) session.
    """
    if i < 200:
        return None
    last = df.iloc[i]
    if pd.isna(last["ATR"]) or last["ATR"] <= 0:
        return None

    # Get hour in UK time (yfinance returns UTC)
    ts = df.index[i]
    hour_uk = (ts.hour + 1) % 24   # UTC → UK (BST)  (adjust for DST if needed)

    in_london = 7 <= hour_uk <= 10
    in_ny     = 13 <= hour_uk <= 16
    if not (in_london or in_ny):
        return None

    # Previous 12-bar high/low (before current bar)
    prev_high = df["High"].iloc[i-12:i].max()
    prev_low  = df["Low"].iloc[i-12:i].min()

    if last["Close"] > prev_high and last["Close"] > last["EMA50"]:
        return {
            "side": "BUY", "strategy": "Session",
            "atr": last["ATR"],
            "stop": last["Close"] - 1.5 * last["ATR"],
            "target": last["Close"] + 2.0 * last["ATR"],
            "min_hold": 2,
        }
    if last["Close"] < prev_low and last["Close"] < last["EMA50"]:
        return {
            "side": "SELL", "strategy": "Session",
            "atr": last["ATR"],
            "stop": last["Close"] + 1.5 * last["ATR"],
            "target": last["Close"] - 2.0 * last["ATR"],
            "min_hold": 2,
        }
    return None


# ============ STRATEGY 3: H4 Trend Following ============
def signal_h4_trend(df, i):
    """
    Simple trend following: buy pullbacks to EMA50 in strong uptrend.
    Uses daily data with large R:R.
    """
    if i < 200:
        return None
    last = df.iloc[i]
    if pd.isna(last["ATR"]) or last["ATR"] <= 0:
        return None

    # Strong trend: EMA50 well above EMA200
    trend_up   = last["EMA50"] > last["EMA200"] * 1.002 and last["Close"] > last["EMA200"]
    trend_down = last["EMA50"] < last["EMA200"] * 0.998 and last["Close"] < last["EMA200"]

    # Pullback to EMA50
    distance_to_ema50 = abs(last["Close"] - last["EMA50"]) / last["ATR"]

    if trend_up and distance_to_ema50 < 0.8 and last["RSI"] < 60:
        return {
            "side": "BUY", "strategy": "Trend",
            "atr": last["ATR"],
            "stop": last["Close"] - 2.0 * last["ATR"],
            "target": last["Close"] + 4.0 * last["ATR"],
            "min_hold": 3,
        }
    if trend_down and distance_to_ema50 < 0.8 and last["RSI"] > 40:
        return {
            "side": "SELL", "strategy": "Trend",
            "atr": last["ATR"],
            "stop": last["Close"] + 2.0 * last["ATR"],
            "target": last["Close"] - 4.0 * last["ATR"],
            "min_hold": 3,
        }
    return None