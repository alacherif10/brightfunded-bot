import pandas as pd
import numpy as np


PAIRS = [
    # (Leg A, Leg B, Hedge ratio approx)
    ("EURUSD=X", "GBPUSD=X"),
    ("AUDUSD=X", "NZDUSD=X"),
    ("AUDUSD=X", "USDCAD=X"),
]


def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

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
    return df


def compute_spread_zscore(df_a, df_b, lookback=60):
    """
    Compute Z-score of the spread between two correlated pairs.
    Spread = log(A) - log(B)  (log ratio, more stable than raw diff)
    Z = (spread - rolling_mean) / rolling_std
    """
    a = df_a["Close"].reindex(df_b.index, method="ffill")
    b = df_b["Close"]
    spread = np.log(a) - np.log(b)

    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    zscore = (spread - mean) / std
    return spread, zscore


def signal_pairs(df_a, df_b, i, z_entry=2.0, z_exit=0.0):
    """
    At bar i, check the Z-score of the log-spread between A and B.
    If Z > z_entry: A is overpriced vs B → SELL A, BUY B
    If Z < -z_entry: A is underpriced vs B → BUY A, SELL B
    """
    if i < 60:
        return None

    spread, z = compute_spread_zscore(df_a.iloc[:i+1], df_b.iloc[:i+1])
    if len(z) < 2 or pd.isna(z.iloc[-1]):
        return None

    z_now = z.iloc[-1]
    atr_a = df_a["ATR"].iloc[i]
    atr_b = df_b["ATR"].iloc[i]

    if pd.isna(atr_a) or pd.isna(atr_b) or atr_a <= 0 or atr_b <= 0:
        return None

    if z_now > z_entry:
        return {
            "side_a": "SELL", "side_b": "BUY",
            "z": z_now, "atr_a": atr_a, "atr_b": atr_b,
        }
    if z_now < -z_entry:
        return {
            "side_a": "BUY", "side_b": "SELL",
            "z": z_now, "atr_a": atr_a, "atr_b": atr_b,
        }
    return None
