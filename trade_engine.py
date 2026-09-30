import json
import os
from datetime import datetime

# ============ CONFIG ============
ACCOUNT_SIZE = 1000.0
DAILY_LOSS_PCT = 0.03
MAX_DD_PCT = 0.06
TARGET_PCT = 0.10
RISK_PER_TRADE_PCT = 0.01
STATE_FILE = "account_state.json"
# ================================


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "initial_balance": ACCOUNT_SIZE,
        "equity": ACCOUNT_SIZE,
        "daily_start_equity": ACCOUNT_SIZE,
        "high_water_mark": ACCOUNT_SIZE,
        "trailing_locked": False,
        "last_date": None,
        "trading_days": [],
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def reset_daily_if_new_day(state):
    today = datetime.now().strftime("%Y-%m-%d")
    if state["last_date"] != today:
        state["daily_start_equity"] = state["equity"]
        state["last_date"] = today
        if today not in state["trading_days"]:
            state["trading_days"].append(today)


def update_state_after_trade(state, pnl_dollars):
    state["equity"] += pnl_dollars
    state["high_water_mark"] = max(state["high_water_mark"], state["equity"])
    if state["equity"] >= state["initial_balance"] + ACCOUNT_SIZE * MAX_DD_PCT:
        state["trailing_locked"] = True


def get_risk_status(state):
    if state["trailing_locked"]:
        floor = state["initial_balance"]
    else:
        floor = max(
            state["high_water_mark"] - ACCOUNT_SIZE * MAX_DD_PCT,
            state["initial_balance"] - ACCOUNT_SIZE * MAX_DD_PCT,
        )
    daily_pnl = state["equity"] - state["daily_start_equity"]
    total_pnl = state["equity"] - state["initial_balance"]
    daily_limit = ACCOUNT_SIZE * DAILY_LOSS_PCT
    remaining_daily = max(0, daily_limit + daily_pnl - 5)
    remaining_total = max(0, state["equity"] - floor - 5)
    return {
        "equity": state["equity"],
        "daily_pnl": daily_pnl,
        "total_pnl": total_pnl,
        "drawdown_floor": floor,
        "remaining_daily_risk": remaining_daily,
        "remaining_total_risk": remaining_total,
        "target_reached": total_pnl >= ACCOUNT_SIZE * TARGET_PCT,
        "halted": (daily_pnl <= -daily_limit) or (state["equity"] <= floor),
        "trading_days": len(state["trading_days"]),
    }


def calculate_lot_size(equity, stop_distance, pip_value_per_lot=10.0, pip_size=0.0001):
    risk_dollars = equity * RISK_PER_TRADE_PCT
    stop_pips = stop_distance / pip_size
    if stop_pips <= 0:
        return 0.0
    lots = risk_dollars / (stop_pips * pip_value_per_lot)
    return round(max(0.01, lots), 2)