import pandas as pd
import numpy as np


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
    df["dist_from_mean"] = (df["Close"] - df["EMA20"]) / df["ATR"]
    return df


def signal_mean_reversion_loose(df, i):
    """Loosened mean reversion — more trades, still has edge."""
    if i < 200:
        return None
    last = df.iloc[i]
    if pd.isna(last["ATR"]) or last["ATR"] <= 0:
        return None

    # Loosened regime filter
    ema_spread = abs(last["EMA50"] - last["EMA200"]) / last["ATR"]
    if ema_spread > 4.0:
        return None

    dist = last["dist_from_mean"]

    # Loosened oversold
    if dist < -1.5 and last["RSI"] < 35:
        stop_dist = 1.5 * last["ATR"]
        return {
            "side": "BUY", "strategy": "MeanRevLoose",
            "atr": last["ATR"],
            "stop": last["Close"] - stop_dist,
            "target": last["EMA20"] + 0.5 * last["ATR"],  # aim slightly past mean
            "min_hold": 3,
        }

    # Loosened overbought
    if dist > 1.5 and last["RSI"] > 65:
        stop_dist = 1.5 * last["ATR"]
        return {
            "side": "SELL", "strategy": "MeanRevLoose",
            "atr": last["ATR"],
            "stop": last["Close"] + stop_dist,
            "target": last["EMA20"] - 0.5 * last["ATR"],
            "min_hold": 3,
        }
    return None
