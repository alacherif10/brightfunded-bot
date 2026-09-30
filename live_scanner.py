import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timezone

# ============================================================
#  CONFIG
# ============================================================
ACCOUNT_EQUITY = 1000.0
RISK_DOLLARS = 15.0

PAIRS = {
    "EURUSD=X": {"name": "EURUSD", "pip": 0.0001, "pip_value": 10.0, "spread": 1.0},
    "GBPUSD=X": {"name": "GBPUSD", "pip": 0.0001, "pip_value": 10.0, "spread": 1.2},
    "EURJPY=X": {"name": "EURJPY", "pip": 0.01,   "pip_value": 6.7,  "spread": 1.5},
}

ALLOWED_HOURS = {9, 10}

# State storage
STATE_FILE = "live_signals.json"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)

# ============================================================
#  TELEGRAM
# ============================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[TELEGRAM - no creds] {message}")
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"[TELEGRAM] Failed: {r.status_code} {r.text}")
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


# ============================================================
#  STATE PERSISTENCE
# ============================================================
_supabase_client = None


def _get_supabase():
    global _supabase_client
    if _supabase_client is None and USE_SUPABASE:
        try:
            from supabase import create_client
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"Supabase init failed: {e}")
    return _supabase_client


def load_state():
    if USE_SUPABASE:
        try:
            sb = _get_supabase()
            if sb:
                response = sb.table("bot_state").select("alerted").eq("id", 1).execute()
                if response.data:
                    return {"alerted": response.data[0]["alerted"] or {}}
        except Exception as e:
            print(f"Supabase load error: {e}")

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"alerted": {}}


def save_state(state):
    if USE_SUPABASE:
        try:
            sb = _get_supabase()
            if sb:
                sb.table("bot_state").update({"alerted": state["alerted"]}).eq("id", 1).execute()
                return
        except Exception as e:
            print(f"Supabase save error: {e}")

    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"File save error: {e}")


# ============================================================
#  HELPERS
# ============================================================
def get_uk_hour(ts):
    month = ts.month
    offset = 1 if 4 <= month <= 10 else 0
    return (ts.hour + offset) % 24


def add_atr(df, period=14):
    df = df.copy()
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    return df


# ============================================================
#  SIGNAL DETECTION
# ============================================================
def check_pair(ticker, meta):
    print(f"  Checking {meta['name']}...")
    try:
        df = yf.download(ticker, period="10d", interval="1h",
                         progress=False, auto_adjust=False)
    except Exception as e:
        print(f"    Download error: {e}")
        return None

    if df is None or df.empty:
        print(f"    No data returned")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if len(df) < 30:
        print(f"    Not enough bars ({len(df)})")
        return None

    df = add_atr(df)
    ts = df.index[-1]
    if ts.tz is not None:
        ts = ts.tz_convert(None)

    hour_uk = get_uk_hour(ts)
    if hour_uk not in ALLOWED_HOURS:
        print(f"    Not London window (hour UK={hour_uk})")
        return None

    today = ts.date()
    idx = [k for k in range(max(0, len(df) - 12), len(df) - 1)
           if df.index[k].date() == today and 0 <= get_uk_hour(df.index[k]) < 7]

    if len(idx) < 3:
        print(f"    Not enough Asian-session bars ({len(idx)})")
        return None

    rh = df["High"].iloc[idx].max()
    rl = df["Low"].iloc[idx].min()
    rw = rh - rl
    atr = df["ATR"].iloc[-1]

    if pd.isna(atr) or atr <= 0:
        print(f"    Invalid ATR")
        return None

    ratio = rw / atr
    if ratio < 0.10 or ratio > 1.5:
        print(f"    Range filter failed (rw/atr={ratio:.2f})")
        return None

    close = float(df["Close"].iloc[-1])

    if close > rh:
        side = "BUY"
        stop = float(rl)
    elif close < rl:
        side = "SELL"
        stop = float(rh)
    else:
        print(f"    No breakout (close inside range)")
        return None

    entry = close
    stop_dist = abs(entry - stop)
    stop_pips = stop_dist / meta["pip"]

    if stop_pips <= 0:
        return None

    lots = RISK_DOLLARS / (stop_pips * meta["pip_value"])
    lots = max(0.01, round(lots, 2))

    return {
        "pair": meta["name"],
        "side": side,
        "entry": entry,
        "stop": stop,
        "range_high": float(rh),
        "range_low": float(rl),
        "range_width": float(rw),
        "stop_pips": float(stop_pips),
        "lots": float(lots),
        "date": str(today),
        "hour_uk": hour_uk,
    }


def format_alert(sig):
    arrow = "⬆️" if sig["side"] == "BUY" else "⬇️"
    return (
        f"{arrow} *{sig['side']} {sig['pair']}*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Entry:     `{sig['entry']:.5f}`\n"
        f"Stop Loss: `{sig['stop']:.5f}`  ({sig['stop_pips']:.1f} pips)\n"
        f"Trail:     0.5 × Asian range (`{sig['range_width']:.5f}`)\n"
        f"Lots:      *{sig['lots']}*\n"
        f"Risk:      ${RISK_DOLLARS:.2f}\n"
        f"UK hour:   {sig['hour_uk']}:00\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"_Trail stop as price moves in your favor._\n"
        f"_Max hold: 30 hours. Timeout close at market._"
    )


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 60)
    print(f"  LIVE SCANNER — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    state = load_state()
    today = str(datetime.now(timezone.utc).date())
    fired = 0

    for ticker, meta in PAIRS.items():
        if state["alerted"].get(meta["name"]) == today:
            print(f"  {meta['name']}: already alerted today, skipping")
            continue

        try:
            sig = check_pair(ticker, meta)
            if sig:
                msg = format_alert(sig)
                ok = send_telegram(msg)
                print(f"    SIGNAL: {sig['side']} {sig['pair']} @ {sig['entry']:.5f}")
                print(f"    Telegram sent: {ok}")
                state["alerted"][meta["name"]] = today
                fired += 1
        except Exception as e:
            print(f"  {meta['name']}: ERROR {e}")

    save_state(state)
    print(f"\n  Alerts sent this run: {fired}")
    print("=" * 60)


if __name__ == "__main__":
    main()