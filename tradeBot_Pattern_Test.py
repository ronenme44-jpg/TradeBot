"""Training/test runner for the reduced public pattern set.

This file mirrors the main bot structure while focusing on repeated signal
evaluation, session-aware scheduling, and reinforcement-score updates.

Typical local flow:
1. Run generate_shared_candles_stocks.py to create shared_candles.csv.
2. Run this file to train/test the reduced ten-pattern strategy set.
3. Runtime JSON/JSONL files are generated automatically and ignored by Git.
"""

#****************************************************************************************************************
#****************************************************************************************************************
# === Step 1: Import libraries and global settings ===

import pandas as pd
import numpy as np
import json
import os
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from sklearn.linear_model import LinearRegression
from config import FEE_BUY as CONFIG_FEE_BUY, FEE_SELL as CONFIG_FEE_SELL

import reinforcement_manager
# === Global paths ===
BASE_DIR = Path(__file__).resolve().parent


def _res(name: str) -> str:
    """Resolve runtime files relative to this repository directory."""
    return str(BASE_DIR / name)


# Runtime files. These are local training state files and are intentionally ignored by Git.
CSV_PATH = _res("shared_candles.csv")
CAPITAL_PATH = _res("BotTest_capital.json")
REINFORCEMENT_PATH = _res("BotTest_reinforcement_scores.json")
PENDING_PATTERN_PATH = _res("BotTest_pending_pattern.json")
CLOSED_LOG_PATH = _res("BotTest_closed_trades_log.json")
SYMBOL = "AAPL"  # US equity symbol (aligned with generate_shared_candles_stocks.py)
BOT_ID = "Bot_Test"

DEFAULT_CAPITAL = 40000
DEFAULT_CAPITAL_PAYLOAD = {"capital": DEFAULT_CAPITAL}
DEFAULT_REINFORCEMENT_PAYLOAD = {}
DEFAULT_PENDING_PATTERN = []
DEFAULT_CLOSED_LOG = []

# --- Resilience / alerting config ---
# Alerts are written to JSONL so data issues can be reviewed after the run.
ALERT_LOG_PATH = _res(f"{BOT_ID}_alerts.jsonl")
ALERT_THROTTLE_SECONDS = 120

# Free market-data feeds can be delayed. The training bot includes a post-close
# grace window so delayed candles can still be processed after 16:00 NY.
DATA_STALE_MINUTES = 20
FREE_DATA_DELAY_MINUTES = DATA_STALE_MINUTES
DATA_GAP_MULTIPLIER = 3.0
MAX_SAME_CANDLE_STREAK = 3
NO_CANDLE_WARN_MULTIPLIER = 2.0

# Small missing gaps can be filled with flat candles. Larger gaps block entries.
FILL_MISSING_CANDLES = True
MAX_MISSING_CANDLES_PER_GAP = 2
MAX_MISSING_CANDLES_PER_SESSION = 3

# The loop polls often, but signal detection happens only when a new candle appears.
POLL_INTERVAL_SECONDS = 5
MAX_LOOP_ERROR_STREAK = 5

_last_alert_times: dict[str, float] = {}

def log_alert(
    level: str,
    message: str,
    *,
    key: str | None = None,
    details: dict | None = None,
    throttle_seconds: int | None = None,
) -> None:
    """Emit a throttled alert to stdout and append to the alerts JSONL log."""
    level_norm = str(level or "INFO").upper()
    throttle = ALERT_THROTTLE_SECONDS if throttle_seconds is None else max(0, int(throttle_seconds))
    key = key or f"{level_norm}:{message}"
    now = time.time()
    last = _last_alert_times.get(key)
    if last is not None and throttle and (now - last) < throttle:
        return
    _last_alert_times[key] = now

    prefix = {"ERROR": "ERROR", "WARN": "WARN", "INFO": "INFO"}.get(level_norm, "INFO")
    print(f"{prefix} {message}")

    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "level": level_norm,
        "message": message,
        "key": key,
    }
    if isinstance(details, dict):
        payload["details"] = details
    try:
        with open(ALERT_LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def log_exception(message: str, exc: Exception, *, key: str) -> None:
    """Log an exception with a short traceback payload (best effort)."""
    try:
        tb = traceback.format_exc()
    except Exception:
        tb = None
    details = {"error": str(exc)}
    if tb:
        details["traceback"] = tb
    log_alert("ERROR", message, key=key, details=details)


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        val = float(value)
        if not np.isfinite(val):
            return default
        return val
    except Exception:
        return default

reinforcement_manager.REINFORCEMENT_PATH = REINFORCEMENT_PATH
# Alias to avoid NameError in any scope that refers to load_scores
load_scores = reinforcement_manager.load_scores

# --- Project modules ---
from tradeBot_Thresholds import THRESHOLDS
from tradeBot_Graphical_Pattern import detect_Graphical_patterns
from tradeBot_indicators_pattern import confirmation_functions
from reinforcement_manager import update_score_on_close, get_score, load_scores as rm_load_scores
from tradeBot_Indicator_Rules import pattern_indicator_rules, ensure_reinforcement_key
from terminal_display import render_status as render_snapshot

# === Global parameters ===
# Base TP/SL apply before the ladder reaches its first active threshold.
TP_BASE = 0.01    # Initial take-profit target: 1%
SL_BASE = -0.005  # Initial stop-loss: -0.5%
INTERVAL_MINUTES = 1  # Candle interval in minutes
CAPITAL_PER_TRADE = 40000.0  # Each entry gets an independent 40,000$ budget

# --- Side balancing guard (reduce short over-entry) ---
# Require a higher minimum number of passing indicators for SHORT vs LONG.
# This corrects a tendency to over-trigger shorts without rewriting all detectors.
MIN_TRUE_INDICATORS_LONG = 2
MIN_TRUE_INDICATORS_SHORT = 3

# Feature toggles for entry filters
ENABLE_TREND_FILTERS = False   # price vs EMA200 + slope; off in the public default configuration
ENABLE_VOL_FILTER    = False   # ATR/close band; off in the public default configuration
ENABLE_SPREAD_FILTER = False   # EMA50/200 spread band; off in the public default configuration
# Verbose entry log with pattern + indicators
SHOW_PATTERN_NAME_LOGS = True

# Risk per trade (fraction of capital)
RISK_PCT_LOW  = 0.005  # 0.5%
RISK_PCT_HIGH = 0.010  # 1.0%

# Market regime filters
VOL_MIN = 0.003   # 0.30% ATR/close minimum
VOL_MAX = 0.020   # 2.00% ATR/close maximum
SPREAD_MIN = 0.002  # 0.20% |ema50-ema200|/close minimum
SPREAD_MAX = 0.030  # 3.00% maximum
EMA_SLOPE_WINDOW = 20  # bars to estimate ema200 slope sign
TREND_BAND_PCT = 0.0005  # 0.05% neutral band around EMA200 for trend filter

# Market session (US equities)
# The machine can run in any local timezone. All market decisions use New York
# time, while Israel time is printed for convenience.
MARKET_TZ = ZoneInfo("America/New_York")
LOCAL_TZ = ZoneInfo("Asia/Jerusalem")
MARKET_OPEN_MINUTE = 9 * 60 + 30   # 09:30 NY
MARKET_CLOSE_MINUTE = 16 * 60      # 16:00 NY
POST_CLOSE_GRACE_SECONDS = (FREE_DATA_DELAY_MINUTES + 1) * 60  # delayed free data + 1m after close
POST_CLOSE_POLL_SECONDS = 60
FORCED_EXIT_BUFFER_MIN = 10        # force-close open trades this many minutes before close
ENTRY_BLOCK_BUFFER_MIN = 70        # block new entries 70 minutes before close

# Simple percent formatter (terminal utility)
def format_percent(x: float) -> str:
    """Return percentage string like '+0.60%'. Input is a fraction (0.006 = 0.6%)."""
    try:
        return f"{x*100:+.2f}%"
    except Exception:
        return "+0.00%"


# --- Market session helpers (US equities regular hours) ---
def _market_time_on_date(ny_dt: datetime, minute_of_day: int) -> datetime:
    return ny_dt.replace(
        hour=minute_of_day // 60,
        minute=minute_of_day % 60,
        second=0,
        microsecond=0,
    )


def _next_regular_market_open(ny_now: datetime) -> datetime:
    """Return the next regular US equity open in New York time (weekday schedule)."""
    candidate = _market_time_on_date(ny_now, MARKET_OPEN_MINUTE)
    if ny_now.weekday() < 5 and ny_now < candidate:
        return candidate

    days_ahead = 1
    while True:
        next_day = ny_now + timedelta(days=days_ahead)
        if next_day.weekday() < 5:
            return _market_time_on_date(next_day, MARKET_OPEN_MINUTE)
        days_ahead += 1


def _format_sleep_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def market_session_status(now_utc: datetime | None = None) -> dict:
    """Return session flags/timestamps for US market hours, with Israel local time for logging."""
    now_utc = now_utc or datetime.now(timezone.utc)
    ny_now = now_utc.astimezone(MARKET_TZ)
    il_now = now_utc.astimezone(LOCAL_TZ)

    open_dt = _market_time_on_date(ny_now, MARKET_OPEN_MINUTE)
    close_dt = _market_time_on_date(ny_now, MARKET_CLOSE_MINUTE)
    sleep_after_close_dt = close_dt + timedelta(seconds=POST_CLOSE_GRACE_SECONDS)

    is_weekday = ny_now.weekday() < 5
    is_open = bool(is_weekday and open_dt <= ny_now < close_dt)
    is_post_close_grace = bool(is_weekday and close_dt <= ny_now < sleep_after_close_dt)
    is_active_window = bool(is_open or is_post_close_grace)
    next_open_dt = None if is_active_window else _next_regular_market_open(ny_now)
    seconds_to_next_open = (
        max(0.0, (next_open_dt - ny_now).total_seconds())
        if next_open_dt is not None
        else 0.0
    )
    minutes_to_close = (close_dt - ny_now).total_seconds() / 60 if is_weekday else None
    minutes_from_open = (ny_now - open_dt).total_seconds() / 60 if is_weekday else None

    return {
        "now_utc": now_utc,
        "now_ny": ny_now,
        "now_il": il_now,
        "is_weekday": is_weekday,
        "is_open": is_open,
        "is_post_close_grace": is_post_close_grace,
        "is_active_window": is_active_window,
        "open_dt": open_dt,
        "close_dt": close_dt,
        "sleep_after_close_dt": sleep_after_close_dt,
        "next_open_dt": next_open_dt,
        "seconds_to_next_open": seconds_to_next_open,
        "minutes_to_close": minutes_to_close,
        "minutes_from_open": minutes_from_open,
    }


def sleep_until_next_market_open(session: dict) -> None:
    """Sleep while the US market is closed, waking at the next regular open."""
    now_ny = session.get("now_ny")
    if not isinstance(now_ny, datetime):
        now_ny = datetime.now(timezone.utc).astimezone(MARKET_TZ)
    next_open = session.get("next_open_dt")
    if not isinstance(next_open, datetime):
        next_open = _next_regular_market_open(now_ny)

    seconds_to_open = max(0.0, (next_open - now_ny).total_seconds())
    next_open_il = next_open.astimezone(LOCAL_TZ)
    log_alert(
        "INFO",
        (
            "US market closed. Sleeping until next open: "
            f"NY {next_open.strftime('%Y-%m-%d %H:%M')} | "
            f"Israel {next_open_il.strftime('%Y-%m-%d %H:%M')} "
            f"(~{_format_sleep_duration(seconds_to_open)})."
        ),
        key="market_sleep_until_open",
        throttle_seconds=3600,
    )
    time.sleep(max(1.0, seconds_to_open))


def filter_regular_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular session candles (Mon-Fri, 09:30-16:00 NY time)."""
    if df is None or "timestamp" not in df.columns:
        return df

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    market_ts = ts.dt.tz_convert(MARKET_TZ)
    minutes = market_ts.dt.hour * 60 + market_ts.dt.minute
    mask = (market_ts.dt.weekday < 5) & (minutes >= MARKET_OPEN_MINUTE) & (minutes < MARKET_CLOSE_MINUTE)
    filtered = df.loc[mask].reset_index(drop=True).copy()
    filtered["timestamp"] = ts.loc[mask].reset_index(drop=True)
    return filtered


def fill_small_candle_gaps(
    df: pd.DataFrame,
    *,
    source: str,
    interval_minutes: int,
    max_missing_per_gap: int = MAX_MISSING_CANDLES_PER_GAP,
    max_missing_per_session: int = MAX_MISSING_CANDLES_PER_SESSION,
) -> pd.DataFrame:
    """
    Fill small gaps between consecutive candles using flat synthetic bars (prev close).
    Keeps the bot resilient when a candle is missing inside the window.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return df

    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp")
    clean = clean.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if len(clean) < 2:
        return clean

    interval = max(1.0, float(interval_minutes))
    expected_gap = pd.Timedelta(minutes=interval)
    expected_gap_seconds = expected_gap.total_seconds()

    session_dates = clean["timestamp"].dt.tz_convert(MARKET_TZ).dt.date
    clean["_session_date"] = session_dates

    frames = []
    total_missing = 0
    total_filled = 0
    max_gap_min = 0.0
    skipped_sessions = 0

    for _, group in clean.groupby("_session_date", sort=True):
        group = group.drop(columns=["_session_date"]).sort_values("timestamp").reset_index(drop=True)
        if len(group) < 2:
            frames.append(group)
            continue

        gaps: list[tuple[int, int]] = []
        missing_total = 0
        session_max_gap_min = 0.0

        for idx in range(len(group) - 1):
            cur_ts = group.at[idx, "timestamp"]
            next_ts = group.at[idx + 1, "timestamp"]
            gap_seconds = (next_ts - cur_ts).total_seconds()
            if gap_seconds <= expected_gap_seconds:
                continue
            gap_min = gap_seconds / 60.0
            if gap_min > session_max_gap_min:
                session_max_gap_min = gap_min
            steps = int(round(gap_seconds / expected_gap_seconds))
            missing = max(0, steps - 1)
            if missing == 0:
                continue
            gaps.append((idx, missing))
            missing_total += missing

        total_missing += missing_total
        if session_max_gap_min > max_gap_min:
            max_gap_min = session_max_gap_min

        if not gaps:
            frames.append(group)
            continue

        too_many_missing = missing_total > max_missing_per_session or any(m > max_missing_per_gap for _, m in gaps)
        if too_many_missing:
            skipped_sessions += 1
            frames.append(group)
            continue

        filled_rows = []
        for idx, missing in gaps:
            prev_close = _safe_float(group.at[idx, "close"])
            if prev_close is None or prev_close <= 0:
                prev_close = _safe_float(group.at[idx + 1, "open"]) or _safe_float(group.at[idx + 1, "close"])
            if prev_close is None or prev_close <= 0:
                continue
            base_ts = group.at[idx, "timestamp"]
            for step in range(1, missing + 1):
                filled_rows.append({
                    "timestamp": base_ts + expected_gap * step,
                    "open": prev_close,
                    "high": prev_close,
                    "low": prev_close,
                    "close": prev_close,
                    "volume": 0.0,
                })

        if filled_rows:
            filled_df = pd.DataFrame(filled_rows)
            group = pd.concat([group, filled_df], ignore_index=True)
            group = group.sort_values("timestamp").reset_index(drop=True)
            total_filled += len(filled_rows)

        frames.append(group)

    if total_filled:
        log_alert(
            "WARN",
            f"Filled {total_filled} missing candle(s) in {source}.",
            key=f"fill_missing:{source}",
            details={
                "filled": total_filled,
                "missing_total": total_missing,
                "max_gap_min": round(max_gap_min, 2),
                "skipped_sessions": skipped_sessions,
            },
        )
    elif total_missing:
        log_alert(
            "WARN",
            f"Detected {total_missing} missing candle(s) in {source}; left unfilled.",
            key=f"missing_unfilled:{source}",
            details={
                "missing_total": total_missing,
                "max_gap_min": round(max_gap_min, 2),
                "max_missing_per_gap": max_missing_per_gap,
                "max_missing_per_session": max_missing_per_session,
                "skipped_sessions": skipped_sessions,
            },
        )

    if not frames:
        return clean.drop(columns=["_session_date"]).reset_index(drop=True)

    clean = pd.concat(frames, ignore_index=True)
    return clean.sort_values("timestamp").reset_index(drop=True)


def detect_unfilled_candle_gap_issue(
    df: pd.DataFrame,
    *,
    interval_minutes: int,
    max_missing_per_gap: int = MAX_MISSING_CANDLES_PER_GAP,
    max_missing_per_session: int = MAX_MISSING_CANDLES_PER_SESSION,
) -> dict:
    """Return current-session gap metadata that should block new entries."""
    report = {
        "block_entries": False,
        "missing_total": 0,
        "max_gap_min": 0.0,
        "max_missing_in_gap": 0,
        "max_missing_per_gap": max_missing_per_gap,
        "max_missing_per_session": max_missing_per_session,
    }
    if df is None or df.empty or "timestamp" not in df.columns:
        return report

    clean = df[["timestamp"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp")
    clean = clean.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if len(clean) < 2:
        return report

    clean["_session_date"] = clean["timestamp"].dt.tz_convert(MARKET_TZ).dt.date
    latest_session = clean["_session_date"].iloc[-1]
    session_ts = clean.loc[clean["_session_date"] == latest_session, "timestamp"].reset_index(drop=True)
    if len(session_ts) < 2:
        report["session"] = str(latest_session)
        return report

    expected_gap_seconds = pd.Timedelta(minutes=max(1.0, float(interval_minutes))).total_seconds()
    missing_total = 0
    max_gap_min = 0.0
    max_missing_in_gap = 0

    for idx in range(len(session_ts) - 1):
        gap_seconds = (session_ts.iloc[idx + 1] - session_ts.iloc[idx]).total_seconds()
        if gap_seconds <= expected_gap_seconds:
            continue
        steps = int(round(gap_seconds / expected_gap_seconds))
        missing = max(0, steps - 1)
        if missing == 0:
            continue
        missing_total += missing
        max_missing_in_gap = max(max_missing_in_gap, missing)
        max_gap_min = max(max_gap_min, gap_seconds / 60.0)

    report.update({
        "session": str(latest_session),
        "missing_total": int(missing_total),
        "max_gap_min": round(max_gap_min, 2),
        "max_missing_in_gap": int(max_missing_in_gap),
        "block_entries": bool(
            missing_total > max_missing_per_session
            or max_missing_in_gap > max_missing_per_gap
        ),
    })
    return report


def compute_data_delay_minutes(last_ts: pd.Timestamp | None, now_utc: datetime | None = None) -> float | None:
    """Return delay between now and last candle (minutes)."""
    if last_ts is None or pd.isna(last_ts):
        return None
    now_utc = now_utc or datetime.now(timezone.utc)
    try:
        last_dt = pd.to_datetime(last_ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        else:
            last_dt = last_dt.tz_convert(timezone.utc)
        delta = now_utc - last_dt
        return delta.total_seconds() / 60.0
    except Exception:
        return None


# === Pretty trade summary helpers ===
def _fmt_usd(x: float) -> str:
    """Return signed dollar string like +$1.23 / -$0.45"""
    return f"{'+' if x >= 0 else '-'}${abs(x):.2f}"

def print_trade_summary(open_trade: dict, exit_price: float, exit_ts: str) -> None:
    """
    Pretty, detailed end-of-trade summary.
    Shows gross vs. net, fees breakdown, and ladder level.
    """
    side = open_trade.get("signal", "LONG")
    entry = float(open_trade.get("entry_price", 0.0))
    qty = float(open_trade.get("quantity", 0.0))
    entry_ts = open_trade.get("entry_time", "N/A")
    level = int(open_trade.get("current_level", 0))

    # Gross P&L (before fees)
    gross_frac = compute_profit_move_fraction(entry, exit_price, side)  # + when profitable
    if side == "LONG":
        gross_usd = (exit_price - entry) * qty
    else:
        gross_usd = (entry - exit_price) * qty

    # Fees (percent-of-notional at entry and exit)
    fee_buy = float(globals().get("FEE_BUY", 0.0))
    fee_sell = float(globals().get("FEE_SELL", 0.0))
    buy_fee_usd = entry * qty * fee_buy
    sell_fee_usd = exit_price * qty * fee_sell
    total_fees_usd = buy_fee_usd + sell_fee_usd

    # Net P&L (use the canonical net calculators with configured fees)
    if side == "LONG":
        net_frac = calc_net_pnl_fraction_long(entry, exit_price, fee_buy, fee_sell)
    else:
        net_frac = calc_net_pnl_fraction_short(entry, exit_price, fee_sell, fee_buy)
    net_usd = entry * qty * net_frac

    # Pretty print
    print("--------------------------------------------------------")
    outcome = "success" if net_usd >= 0 else "fail"
    print(f"Trade Summary {side}")
    print(f"Qty: {int(qty)} | Levels: {level}")
    print(f"Entry: {entry:.2f} @ {entry_ts}")
    print(f"Exit : {exit_price:.2f} @ {exit_ts} (reason: SL hit)")
    print(f"Gross: {format_percent(gross_frac)} ({_fmt_usd(gross_usd)})")
    print(f"Fees : {_fmt_usd(-buy_fee_usd)} entry + {_fmt_usd(-sell_fee_usd)} exit = {_fmt_usd(-total_fees_usd)} total")
    print(f"Net  : {format_percent(net_frac)} ({_fmt_usd(net_usd)})  -> {outcome}")
    print("--------------------------------------------------------")

# Public strategy set. Names must match:
# - tradeBot_Graphical_Pattern.py detector registry
# - tradeBot_indicators_pattern.py confirmation_functions
# - tradeBot_Indicator_Rules.py pattern_indicator_rules
pattern_list = [
    {"name": "bullish_engulfing", "type": "LONG"},
    {"name": "hammer", "type": "LONG"},
    {"name": "morning_star", "type": "LONG"},
    {"name": "piercing_pattern", "type": "LONG"},
    {"name": "three_white_soldiers_simple", "type": "LONG"},
    {"name": "bearish_engulfing", "type": "SHORT"},
    {"name": "shooting_star", "type": "SHORT"},
    {"name": "evening_star", "type": "SHORT"},
    {"name": "three_black_crows", "type": "SHORT"},
    {"name": "dark_cloud_cover", "type": "SHORT"},
]
#****************************************************************************************************************
#****************************************************************************************************************
# Step 2: Load latest candles from CSV

REQUIRED_CANDLE_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def validate_candles_df(df, *, source: str, min_rows: int = 2) -> pd.DataFrame | None:
    """
    Validate and sanitize candle data to avoid crashes downstream.
    Returns a cleaned DataFrame or None if insufficient/invalid.
    """
    if df is None:
        log_alert("WARN", f"No data returned from {source}.", key=f"no_data:{source}")
        return None
    if not isinstance(df, pd.DataFrame):
        log_alert("ERROR", f"Invalid data type from {source}: {type(df)}", key=f"bad_type:{source}")
        return None

    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        log_alert(
            "ERROR",
            f"Missing candle columns from {source}: {', '.join(missing)}",
            key=f"missing_cols:{source}",
        )
        return None

    clean = df[REQUIRED_CANDLE_COLS].copy()
    clean['timestamp'] = pd.to_datetime(clean['timestamp'], utc=True, errors='coerce')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        clean[col] = pd.to_numeric(clean[col], errors='coerce')

    before = len(clean)
    clean = clean.dropna(subset=REQUIRED_CANDLE_COLS)
    dropped = before - len(clean)
    if dropped:
        log_alert(
            "WARN",
            f"Dropped {dropped} invalid candle row(s) from {source}.",
            key=f"drop_rows:{source}",
            details={"dropped": dropped, "before": before},
        )

    if len(clean) == 0:
        return None

    bad_prices = (
        (clean['open'] <= 0) | (clean['high'] <= 0) | (clean['low'] <= 0) | (clean['close'] <= 0)
    )
    bad_prices |= (clean['high'] < clean['low'])
    if bad_prices.any():
        count_bad = int(bad_prices.sum())
        clean = clean.loc[~bad_prices].copy()
        log_alert(
            "WARN",
            f"Removed {count_bad} rows with invalid price ranges from {source}.",
            key=f"bad_prices:{source}",
        )

    if len(clean) == 0:
        return None

    neg_vol = clean['volume'] < 0
    if neg_vol.any():
        count_neg = int(neg_vol.sum())
        clean.loc[neg_vol, 'volume'] = 0.0
        log_alert(
            "WARN",
            f"Clamped {count_neg} negative volume rows to 0 in {source}.",
            key=f"neg_volume:{source}",
        )

    if clean.duplicated(subset=['timestamp']).any():
        count_dup = int(clean.duplicated(subset=['timestamp']).sum())
        clean = clean.drop_duplicates(subset=['timestamp'], keep='last')
        log_alert(
            "WARN",
            f"Removed {count_dup} duplicate timestamps from {source}.",
            key=f"dup_ts:{source}",
        )

    clean = clean.sort_values('timestamp').reset_index(drop=True)
    if len(clean) < max(1, int(min_rows)):
        log_alert(
            "WARN",
            f"Not enough clean candles from {source} (have {len(clean)}, need {min_rows}).",
            key=f"too_few:{source}",
        )
        return None

    return clean

def load_candles_from_csv(path=CSV_PATH, limit=390):
    """
    Load the candle CSV and keep only the latest rows.
    """
    if not os.path.exists(path):
        log_alert("WARN", f"CSV file not found: {path}", key="csv_missing")
        return None

    try:
        df = pd.read_csv(path)

        # Keep only the required columns.
        missing_cols = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
        if missing_cols:
            log_alert("ERROR", f"Missing columns in CSV: {', '.join(missing_cols)}", key="csv_missing_cols")
            return None

        df = df[REQUIRED_CANDLE_COLS].copy()

        # Safe conversions so corrupt rows do not stop the bot.
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop corrupt rows instead of stopping the bot.
        before = len(df)
        df = df.dropna(subset=REQUIRED_CANDLE_COLS).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            log_alert(
                "WARN",
                f"Dropped {dropped} corrupt candle row(s) from CSV.",
                key="csv_drop_rows",
                details={"dropped": dropped, "before": before},
            )

        # Sort by time and keep the latest rows.
        df = df.sort_values('timestamp').tail(limit).reset_index(drop=True)
        df = validate_candles_df(df, source="CSV", min_rows=2)
        if df is None:
            return None
        return df.tail(limit).reset_index(drop=True)

    except Exception as e:
        log_exception("Error loading CSV.", e, key="csv_load")
        return None
# Load latest candles from an optional broker/API source.
def load_candles_from_api(limit=390):
    """
    Placeholder for a future real broker/API integration.

    The public repository never fabricates random candles. If the CSV is missing,
    run generate_shared_candles_stocks.py first.
    """
    log_alert(
        "WARN",
        "No broker/API candle integration is configured. Run generate_shared_candles_stocks.py to create shared_candles.csv.",
        key="api_not_configured",
    )
    return None

#****************************************************************************************************************
#****************************************************************************************************************
# Step 3: Initialize and validate required JSON files

def _backup_corrupt_json(path: str) -> None:
    """Rename a corrupt JSON file to preserve it before reinitializing."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.replace(path, f"{path}.corrupt.{ts}")
    except Exception:
        pass


def safe_read_json(path: str, default_value, *, expect_type=None, purpose: str | None = None):
    """Read JSON safely; if corrupt or missing, return default and reinitialize."""
    if not os.path.exists(path):
        return default_value
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        label = purpose or "json"
        log_alert(
            "ERROR",
            f"{label} file corrupted: {path}. Resetting to defaults.",
            key=f"json_decode:{path}",
            details={"error": str(exc)},
        )
        _backup_corrupt_json(path)
        safe_write_json(path, default_value, purpose=label)
        return default_value
    except Exception as exc:
        label = purpose or "json"
        log_exception(f"Failed to read {label} file: {path}", exc, key=f"json_read:{path}")
        return default_value

    if expect_type and not isinstance(data, expect_type):
        label = purpose or "json"
        log_alert(
            "WARN",
            f"{label} file has unexpected shape; resetting.",
            key=f"json_shape:{path}",
        )
        safe_write_json(path, default_value, purpose=label)
        return default_value
    return data


def safe_write_json(path: str, payload, *, purpose: str | None = None) -> None:
    """Write JSON atomically so partial writes do not corrupt runtime state."""
    try:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception as exc:
        label = purpose or "json"
        log_exception(f"Failed to write {label} file: {path}", exc, key=f"json_write:{path}")

def init_file(path, default_value):
    """
    Create a JSON file if it does not exist.
    """
    if not os.path.exists(path):
        safe_write_json(path, default_value, purpose="init")
        print(f"Created missing file: {path}")

def initialize_all_files():
    """
    Ensure all required JSON files exist with defaults.
    """
    init_file(CAPITAL_PATH, DEFAULT_CAPITAL_PAYLOAD)
    init_file(REINFORCEMENT_PATH, DEFAULT_REINFORCEMENT_PAYLOAD)
    init_file(PENDING_PATTERN_PATH, DEFAULT_PENDING_PATTERN)  # Open trades list.
    # Closed trades log disabled for the training bot.
    # init_file(CLOSED_LOG_PATH, DEFAULT_CLOSED_LOG)

# === CAPITAL ===
def load_capital():
    if not os.path.exists(CAPITAL_PATH):
        log_alert("WARN", "CAPITAL file missing. Reinitializing...", key="capital_missing")
        initialize_all_files()
    payload = safe_read_json(
        CAPITAL_PATH,
        DEFAULT_CAPITAL_PAYLOAD,
        expect_type=dict,
        purpose="capital",
    )
    cap = _safe_float(payload.get("capital"), DEFAULT_CAPITAL) if isinstance(payload, dict) else DEFAULT_CAPITAL
    return cap if cap is not None else DEFAULT_CAPITAL

def save_capital(value):
    safe_write_json(CAPITAL_PATH, {'capital': value}, purpose="capital")


def _strip_internal_fields(trade_data):
    """Remove internal-only keys (prefixed with '_') before persisting logs."""
    if isinstance(trade_data, dict):
        return {k: v for k, v in trade_data.items() if not str(k).startswith("_")}
    return trade_data


# === CLOSED TRADE LOG ===
# Disabled for the training bot. Kept here in comments in case we want to restore it later.
# def update_closed_trade_log(trade_data, path=CLOSED_LOG_PATH):
#     """
#     Save a closed-trade log.
#     """
#     try:
#         logs = safe_read_json(path, DEFAULT_CLOSED_LOG, expect_type=list, purpose="closed_trades")
#         logs.append(_strip_internal_fields(trade_data))
#         safe_write_json(path, logs, purpose="closed_trades")
#     except Exception as e:
#         log_exception("Error writing closed trade log.", e, key="closed_log_write")
#
#
# def count_closed_trades(path=CLOSED_LOG_PATH) -> int:
#     """Return the number of closed trades in the log file."""
#     try:
#         data = safe_read_json(path, DEFAULT_CLOSED_LOG, expect_type=list, purpose="closed_trades")
#         return len(data) if isinstance(data, list) else 0
#     except Exception:
#         return 0

# === PENDING PATTERN ===
def load_pending_pattern():
    """Load open training trades from disk."""
    data = safe_read_json(PENDING_PATTERN_PATH, DEFAULT_PENDING_PATTERN, purpose="pending")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Backward compatibility with the old single-trade format.
        return [data]
    return []

def save_pending_pattern(pattern_data):
    """Persist open training trades; empty means no open trades."""
    payload = pattern_data
    if pattern_data is None:
        payload = []
    safe_write_json(PENDING_PATTERN_PATH, payload, purpose="pending")

def persist_open_trades(open_trades: list[dict]) -> None:
    """Persist the open trades list to disk."""
    safe_trades = []
    for t in open_trades:
        try:
            if t.get("status") != "open":
                continue
            safe_trades.append(_strip_internal_fields(t))
        except Exception:
            continue
    save_pending_pattern(safe_trades)

#****************************************************************************************************************
#****************************************************************************************************************
# Step 4: Trade management (enter / ladder / exit)

# Fees (fractions) for NET PnL calculations (from config)
FEE_BUY  = CONFIG_FEE_BUY
FEE_SELL = CONFIG_FEE_SELL


def get_pattern_side(name: str) -> str:
    """Return the side ('LONG'/'SHORT') for a given pattern name based on pattern_list."""
    for item in pattern_list:
        if item.get("name") == name:
            return item.get("type", "LONG").upper()
    return "LONG"


def calc_net_pnl_fraction_long(entry_price: float, exit_price: float, fee_buy: float, fee_sell: float) -> float:
    """NET PnL for LONG after fees (fraction)."""
    return ((exit_price / entry_price) * (1 - fee_sell) / (1 + fee_buy)) - 1


def calc_net_pnl_fraction_short(entry_sell_price: float, exit_buy_price: float, fee_sell: float, fee_buy: float) -> float:
    """NET PnL for SHORT after fees (fraction)."""
    return ((entry_sell_price / exit_buy_price) * (1 - fee_sell) / (1 + fee_buy)) - 1


def build_indicator_outcomes(pattern_name: str, confirm_result) -> dict:
    """
    Return {indicator_name: bool} aligned with the pattern's indicator schema.
    Supported inputs:
      1) {"overall": bool, "checks": {name: bool, ...}}
      2) flat dict of booleans without overall
      3) simple boolean value; no per-indicator inference, returns {}

    Rules:
    - Only explicitly returned indicators from checks are counted.
    - Missing indicators are mapped to False when an expected list exists.
    - Never infer individual indicators from overall.
    """
    expected = []
    try:
        rules = pattern_indicator_rules.get(pattern_name, [])
        expected = [r.get("indicator") for r in rules if isinstance(r, dict) and r.get("indicator")]
    except Exception:
        expected = []

    if isinstance(confirm_result, dict):
        if "checks" in confirm_result and isinstance(confirm_result["checks"], dict):
            raw = {str(k): bool(v) for k, v in confirm_result["checks"].items()}
        else:
            raw = {str(k): bool(v) for k, v in confirm_result.items()
                   if k != "overall" and not isinstance(v, (dict, list))}
        # Support staged keys like 'pre.rsi' by collapsing to the last token
        raw_simple = {}
        for k, v in raw.items():
            simple = str(k).split('.')[-1]
            raw_simple[simple] = bool(v) or bool(raw_simple.get(simple, False))
        if expected:
            return {name: bool(raw_simple.get(name, False)) for name in expected}
        return raw_simple

    # Simple bool: do not infer per-indicator values.
    return {}

# normalize_confirmation_result is defined once later (staged-aware)

def compute_profit_move_fraction(entry_price: float, current_price: float, side: str) -> float:
    """
    Compute GROSS price move fraction since entry, positive when the trade is in profit.
    - LONG : (current/entry) - 1
    - SHORT: (entry/current) - 1
    This is used for ladder levels only (not for NET PnL).
    """
    if side == "SHORT":
        if current_price <= 0:
            return 0.0
        return (entry_price / current_price) - 1.0
    else:
        return (current_price / entry_price) - 1.0


def calc_base_tp_sl(entry_price: float, side: str) -> tuple:
    """
    Calculate initial TP/SL from the base constants (before ladder starts).
    - LONG : TP at +TP_BASE, SL at SL_BASE (negative) from entry
    - SHORT: TP at -TP_BASE (down), SL at +|SL_BASE| (up) from entry
    Returns (tp_price, sl_price)
    """
    if side == "SHORT":
        tp_price = entry_price * (1 - abs(TP_BASE))
        sl_price = entry_price * (1 + abs(SL_BASE))
    else:
        tp_price = entry_price * (1 + abs(TP_BASE))
        sl_price = entry_price * (1 + SL_BASE)  # SL_BASE is negative
    return tp_price, sl_price


def update_ladder_targets(open_trade: dict, current_price: float) -> dict:
    """
    Update ladder level, TP, and SL based on THRESHOLDS and current price.
    Important: you can only go UP levels (no downgrade). Exit occurs only on SL.

    THRESHOLDS rows are expected like: (level, profit_thr, tp_frac, sl_frac)
    Example for level 1: (1, 0.0015, 0.0035, 0.0012)
    Meaning (for LONG): when profit >= 0.15% -> TP=+0.35%, SL=+0.12% (in profit)
    For SHORT we lock **both** TP and SL **below** the entry price (profit side).
    """
    side = open_trade.get("signal", "LONG")
    entry = open_trade.get("entry_price")
    if entry is None or current_price is None:
        return open_trade

    # Compute gross move since entry (positive when in profit)
    move = compute_profit_move_fraction(entry, current_price, side)

    # Determine highest passed level by move
    tp_f = None
    sl_f = None
    passed_level = 0
    for row in THRESHOLDS:
        lvl, thr, tp_f_row, sl_f_row = row
        if move >= float(thr):
            passed_level = max(passed_level, int(lvl))

    # Enforce no-downgrade rule
    current_level = int(open_trade.get("current_level", 0))
    prev_level = current_level
    new_level = max(current_level, passed_level)
    open_trade["current_level"] = new_level
    level_up = new_level > prev_level

    # Compute TP/SL for the active regime
    if new_level <= 0:
        # Base regime before first threshold is reached
        tp_price, sl_price = calc_base_tp_sl(entry, side)
    else:
        # Determine the active level's tp/sl fractions from THRESHOLDS
        for lvl, _thr, tp_frac, sl_frac in THRESHOLDS:
            if int(lvl) == int(new_level):
                tp_f = float(tp_frac)
                sl_f = float(sl_frac)
                break
        if tp_f is None or sl_f is None:
            # Fallback to base if level not found
            tp_price, sl_price = calc_base_tp_sl(entry, side)
        else:
            # IMPORTANT:
            #  - Fractions in THRESHOLDS are ALWAYS profit distances from ENTRY.
            #  - LONG  -> profit is ABOVE entry (add fraction)
            #  - SHORT -> profit is BELOW entry (subtract fraction)
            if side == "SHORT":
                # For SHORT: lock profit with both TP and SL BELOW entry
                tp_price = entry * (1 - tp_f)
                sl_price = entry * (1 - sl_f)
            else:
                # For LONG: lock profit with both TP and SL ABOVE entry
                tp_price = entry * (1 + tp_f)
                sl_price = entry * (1 + sl_f)

    # --- Profit-lock guarantee (NET): once on ladder, SL must lock net profit after fees ---
    if new_level > 0:
        try:
            fee_buy = float(globals().get("FEE_BUY", 0.0))
            fee_sell = float(globals().get("FEE_SELL", 0.0))
        except Exception:
            fee_buy, fee_sell = 0.0, 0.0

        # Minimal gross move to break even after fees
        try:
            g_be = ((1.0 + fee_buy) / (1.0 - fee_sell)) - 1.0
        except Exception:
            g_be = 0.0
        # Safety margin to cover rounding/slippage
        margin = 0.0003  # 0.03%
        g_req = max(0.0, g_be + margin)

        if side == "SHORT":
            # For SHORT, SL must be at or below entry*(1 - g_req)
            min_price = entry * (1.0 - g_req)
            if not isinstance(sl_price, (int, float)) or sl_price > min_price:
                sl_price = float(min_price)
        else:
            # For LONG, SL must be at or above entry*(1 + g_req)
            min_price = entry * (1.0 + g_req)
            if not isinstance(sl_price, (int, float)) or sl_price < min_price:
                sl_price = float(min_price)

    # Persist back to the trade dict and return
    open_trade["tp_price"] = float(tp_price)
    open_trade["sl_price"] = float(sl_price)
    if level_up:
        sl_profit = compute_profit_move_fraction(entry, sl_price, side)
        print(f"Ladder Ladder -> level {new_level} | TP {tp_price:.4f} | SL {sl_price:.4f} | SL profit now {format_percent(sl_profit)}")
    return open_trade


def enter_trade(df, breakout_idx: int, pattern_type: str, capital_data: dict, live=False, api=None, symbol=SYMBOL, context_key: str | None = None, entry_note: str | None = None) -> dict | None:
    """
    Open a trade using the breakout candle index and pattern name.
    - Decides LONG/SHORT by pattern_list
    - Sets base TP/SL (ladder starts only after threshold 1 is reached)
    - Saves to pending_pattern JSON
    """
    try:
        entry_price = float(df.loc[breakout_idx, 'close'])
    except Exception:
        entry_price = float(df['close'].iloc[-1])
    timestamp = df.loc[breakout_idx, 'timestamp'] if 'timestamp' in df.columns else datetime.utcnow().isoformat()

    side = get_pattern_side(pattern_type)

    # Use the pattern as a context label for reinforcement scores.
    context_key = context_key or pattern_type

    tp_price, sl_price = calc_base_tp_sl(entry_price, side)

    # Optional dynamic SL override from capital_data (e.g., ATR-based fraction)
    try:
        sl_override_frac = capital_data.get("sl_override_frac")
        if isinstance(sl_override_frac, (int, float)) and sl_override_frac > 0:
            if side == "SHORT":
                sl_price = entry_price * (1 + float(sl_override_frac))
            else:
                sl_price = entry_price * (1 - float(sl_override_frac))
    except Exception:
        pass

    # Fixed 40,000$ budget per trade; concurrent trades are not capped here.
    try:
        available_capital = float(capital_data.get("capital", CAPITAL_PER_TRADE))
        if available_capital <= 0:
            available_capital = CAPITAL_PER_TRADE
    except Exception:
        available_capital = CAPITAL_PER_TRADE

    # Position sizing priority:
    # 1) explicit position_size (units)
    # 2) risk-based using risk_pct and SL distance
    # 3) allocation fraction fallback
    entry_fee = float(FEE_BUY if side == "LONG" else FEE_SELL)
    per_unit_cost = entry_price * (1.0 + entry_fee)

    if "position_size" in capital_data:
        quantity = max(1.0, float(capital_data.get("position_size", 1.0)))
    else:
        risk_pct = float(capital_data.get("risk_pct", 0.0))
        alloc_frac = float(capital_data.get("allocation_frac", 1.0))
        alloc_cap = available_capital * max(0.0, min(1.0, alloc_frac))

        if risk_pct > 0:
            try:
                sl_dist = abs((sl_price / entry_price) - 1.0)
                sl_dist = max(sl_dist, 1e-4)
                dollar_risk = alloc_cap * max(0.0, min(1.0, risk_pct))
                qty_est = dollar_risk / (entry_price * sl_dist)
                quantity = max(1.0, float(int(qty_est)))
            except Exception:
                quantity = 1.0
        else:
            qty_est = alloc_cap / per_unit_cost if per_unit_cost > 0 else alloc_cap
            quantity = max(1.0, float(int(qty_est)))

    # Do not exceed the per-trade budget, including fees.
    max_qty_allowed = max(1, int(available_capital // per_unit_cost)) if per_unit_cost > 0 else 1
    try:
        quantity = float(min(int(quantity), max_qty_allowed))
    except Exception:
        quantity = float(max_qty_allowed)
    if quantity < 1:
        quantity = 1.0

    trade = {
        "symbol": symbol,
        "signal": side,
        "pattern": pattern_type,
        "context_key": context_key,
        "entry_time": str(timestamp),
        "entry_price": entry_price,
        "quantity": quantity,
        "status": "open",
        "current_level": 0,
        "tp_price": tp_price,
        "sl_price": sl_price,
    }

    position_notional = entry_price * quantity
    ts_short = str(timestamp)[11:16] if isinstance(timestamp, str) else str(timestamp)
    name_block = f" {pattern_type}" if SHOW_PATTERN_NAME_LOGS else ""
    note_part = f" | {entry_note}" if entry_note else ""
    print(f"{ts_short} | {side:<5}{name_block} | entry {entry_price:.4f} | qty {quantity:g} (approx {position_notional:.2f}$){note_part}")
    return trade


def exit_trade(open_trade: dict, exit_price: float, timestamp: str, symbol: str, live: bool = False) -> dict:
    """
    Close an open trade, compute NET PnL after fees, and update logs.
    Returns the updated trade dict.
    """
    entry_price = open_trade.get("entry_price")
    quantity    = open_trade.get("quantity", 1.0)
    signal      = open_trade.get("signal", "LONG")

    # Trade duration (minutes) for reinforcement stats
    duration_min = None
    try:
        entry_ts = pd.to_datetime(open_trade.get("entry_time"))
        exit_ts = pd.to_datetime(timestamp)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.replace(tzinfo=timezone.utc)
        duration = exit_ts - entry_ts
        duration_min = max(0.0, duration.total_seconds() / 60.0)
    except Exception:
        duration_min = None

    # --- Compute NET PnL fraction after fees ---
    try:
        if signal == "LONG":
            net_pnl_fraction = calc_net_pnl_fraction_long(entry_price, exit_price, FEE_BUY, FEE_SELL)
        else:
            net_pnl_fraction = calc_net_pnl_fraction_short(entry_price, exit_price, FEE_SELL, FEE_BUY)
    except Exception as e:
        print(f"WARN Failed to compute NET PnL: {e}")
        net_pnl_fraction = 0.0

    pnl_dollars = round(net_pnl_fraction * entry_price * quantity, 2)
    success = "success" if net_pnl_fraction > 0 else ("neutral" if net_pnl_fraction == 0 else "fail")

    open_trade.update({
        "exit_price": exit_price,
        "exit_time": timestamp,
        "pnl_pct": net_pnl_fraction,
        "net_profit": pnl_dollars,
        "trade_duration_min": duration_min,
        "result": success,
        "status": "closed",
    })

    # Update reinforcement score if we have a context key
    ctx_key = open_trade.get("context_key")
    if ctx_key:
        try:
            update_score_on_close(
                key=ctx_key,
                net_pnl_fraction=net_pnl_fraction,
                net_profit=pnl_dollars,
                trade_duration_min=duration_min,
                timestamp=timestamp,
                symbol=symbol,
                timeframe=f"{INTERVAL_MINUTES}m",
            )
        except Exception as e:
            print(f"WARN RL update failed: {e}")

    # Closed trades log disabled for the training bot.
    # update_closed_trade_log({**open_trade})

    # --- Persist capital update (always applies; includes fees via NET PnL) ---
    prev_capital = None
    new_capital = None
    try:
        prev_capital = float(load_capital())
        new_capital = round(prev_capital + float(pnl_dollars), 2)
        save_capital(new_capital)
    except Exception as e:
        print(f"WARN Failed to update capital_1.json: {e}")

    mode = "LIVE Trade" if live else "Simulated Trade"
    print(f"\n{mode} closed: {signal}")
    print(f"- Entry Price  : {entry_price:.2f}")
    print(f"- Exit Price   : {exit_price:.2f}")
    print(f"Net Result   : {pnl_dollars:.2f}$ ({format_percent(net_pnl_fraction)})")
    print(f"Result Type  : {success}")
    if prev_capital is not None and new_capital is not None:
        print(f"Capital Update : {prev_capital:.2f}$ -> {new_capital:.2f}$")
    print()

    # Pretty, detailed summary at end of trade (non-fatal on error)
    try:
        print_trade_summary(open_trade, exit_price, timestamp)
    except Exception as e:
        print(f"[print_trade_summary error] {e}")

    return open_trade


def on_tick_update_trade(open_trade: dict, current_price: float, timestamp: str, symbol: str, live: bool = False):
    """
    Update ladder (no downgrade) and check SL hit. If SL is hit, close the trade.
    Return (updated_trade, closed_bool)
    """
    if not open_trade or open_trade.get("status") != "open":
        return open_trade, False

    # Update ladder targets
    update_ladder_targets(open_trade, current_price)

    side = open_trade.get("signal", "LONG")
    sl_price = open_trade.get("sl_price")

    # Exit only on SL, by design
    should_close = False
    if side == "SHORT":
        if current_price >= sl_price:
            should_close = True
    else:
        if current_price <= sl_price:
            should_close = True

    if should_close:
        closed = exit_trade(open_trade, exit_price=current_price, timestamp=timestamp, symbol=symbol, live=live)
        return closed, True

    # Optionally, write a lightweight detailed log snapshot (omitted for brevity)
    return open_trade, False


# ============================================================
# Minimal indicators enrichment (to avoid KeyErrors in detectors)
# ============================================================

def _compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using an EMA-style smoothing. Returns a float Series in [0,100].
    Safe for small inputs; will yield NaNs at the head which is fine for detectors.
    """
    close = pd.to_numeric(close_series, errors='coerce')
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Simple EMA using pandas ewm. Returns float Series."""
    s = pd.to_numeric(series, errors='coerce')
    return s.ewm(span=period, adjust=False).mean()

def _slope_sign(series: pd.Series, window: int = 20) -> int:
    """Return +1 if linear slope > 0, -1 if < 0, else 0 over last 'window' points."""
    try:
        tail = pd.to_numeric(series, errors='coerce').tail(max(3, int(window)))
        if len(tail) < 3 or tail.isna().all():
            return 0
        y = tail.values.reshape(-1, 1)
        x = np.arange(len(tail)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        slope = float(model.coef_.ravel()[0])
        if slope > 0:
            return 1
        if slope < 0:
            return -1
        return 0
    except Exception:
        try:
            tail = pd.to_numeric(series, errors='coerce').tail(max(3, int(window)))
            if len(tail) < 3:
                return 0
            slope = float(tail.iloc[-1] - tail.iloc[0])
            return 1 if slope > 0 else (-1 if slope < 0 else 0)
        except Exception:
            return 0


def ensure_min_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure minimal indicator columns exist for detectors/confirmers that expect them.

    Currently guarantees at least:
    - 'rsi_14'
    - EMAs on close: 'ema_9', 'ema_21', 'ema_50', 'ema_200'

    If your detectors raise KeyError for more columns (e.g., SMA volume, ATR, BB),
    add them here in the same pattern.
    """
    # RSI(14)
    if 'rsi_14' not in df.columns:
        try:
            df['rsi_14'] = _compute_rsi(df['close'], period=14)
        except Exception as e:
            print(f"WARN Failed to compute rsi_14: {e}")

    # Common EMAs used by channel/crossover rules
    # NOTE: include 150 because some detectors/confirmers (e.g., golden_cross, some LONG confirmers) rely on ema_150
    ema_periods = [9, 21, 50, 150, 200]
    for p in ema_periods:
        col = f'ema_{p}'
        if col not in df.columns:
            try:
                df[col] = _compute_ema(df['close'], period=p)
            except Exception as e:
                print(f"WARN Failed to compute {col}: {e}")

    # Volume SMA(20) for volume spike checks
    if 'volume_sma_20' not in df.columns:
        try:
            df['volume_sma_20'] = pd.to_numeric(df['volume'], errors='coerce').rolling(window=20, min_periods=1).mean()
        except Exception as e:
            print(f"WARN Failed to compute volume_sma_20: {e}")

    # Close-to-close volatility proxy: rolling std of returns (20)
    if 'volatility_20' not in df.columns:
        try:
            ret = pd.to_numeric(df['close'], errors='coerce').pct_change()
            df['volatility_20'] = ret.rolling(window=20, min_periods=5).std()
        except Exception as e:
            print(f"WARN Failed to compute volatility_20: {e}")

    # ATR(14) for range/volatility-aware rules
    if 'atr_14' not in df.columns:
        try:
            high = pd.to_numeric(df['high'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')
            close = pd.to_numeric(df['close'], errors='coerce')
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(window=14, min_periods=5).mean()
        except Exception as e:
            print(f"WARN Failed to compute atr_14: {e}")

    # Backward-compat aliases used by some confirmers
    if 'volatility' not in df.columns and 'volatility_20' in df.columns:
        try:
            df['volatility'] = df['volatility_20']
        except Exception:
            pass

    if 'mean_volume' not in df.columns:
        try:
            df['mean_volume'] = pd.to_numeric(df['volume'], errors='coerce').rolling(window=10, min_periods=1).mean()
        except Exception:
            pass

    return df

# ============================================================
# Heartbeat printer - one line status every loop
# ============================================================
def print_heartbeat(
    current_price: float,
    open_trades: list[dict],
    new_trades: list[dict] | None = None,
    *,
    session: dict | None = None,
    data_delay_min: float | None = None,
    allow_new_entries: bool = True,
):
    """
    Print a compact status line: time, price, new trades, open trades, and session state.
    """
    now_str = datetime.now(timezone.utc).strftime("%H:%M")
    session_note = ""
    if isinstance(session, dict):
        ny_now = session.get("now_ny")
        il_now = session.get("now_il")
        is_open = bool(session.get("is_open"))
        mtc = session.get("minutes_to_close")
        if isinstance(ny_now, datetime):
            now_str = ny_now.strftime("%H:%M NY")
        ny_txt = ny_now.strftime("%H:%M") if isinstance(ny_now, datetime) else "n/a"
        il_txt = il_now.strftime("%H:%M") if isinstance(il_now, datetime) else "n/a"
        if is_open:
            session_note = f"NY {ny_txt}"
            if mtc is not None:
                session_note += f" ({mtc:.0f}m to close)"
            session_note += f" | IL {il_txt}"
        else:
            session_note = f"Market closed | NY {ny_txt} | IL {il_txt}"

    new_count = len(new_trades or [])
    # Closed trades log disabled for the training bot.
    # closed_total = count_closed_trades()

    parts = [
        f"Time {now_str}",
        f"price {current_price:.4f}",
        f"new {new_count}",
        f"open {len(open_trades)}",
        # f"closed total {closed_total}",
    ]
    if not allow_new_entries:
        parts.append(f"gate closed >={ENTRY_BLOCK_BUFFER_MIN}m to close")
    if data_delay_min is not None:
        parts.append(f"delay {data_delay_min:.1f}m")
    if session_note:
        parts.append(session_note)

    print(" | ".join(parts))

# ============================================================
# Glue - detect -> confirm -> key -> ensure -> enter trade
# ============================================================


# --- Helper: normalize confirmation results from confirmers (staged-aware) ---
def normalize_confirmation_result(name: str, df: pd.DataFrame, idx: int):
    """
    Normalize confirmer output to (overall_bool_or_None, checks_dict).
    Supports: bool, (overall, checks), {overall, checks}, {overall, stages:{...}},
    or a flat dict of indicator booleans.
    """
    confirmer = confirmation_functions.get(name)
    if confirmer is None:
        return None, {}
    try:
        res = confirmer(df, idx)
    except Exception as e:
        try:
            cname = getattr(confirmer, '__name__', str(confirmer))
        except Exception:
            cname = 'confirmer'
        print(f"ERROR in {cname}: {e}")
        return None, {}

    # bool
    if isinstance(res, bool):
        return res, {}

    # tuple(overall, checks)
    if isinstance(res, tuple) and len(res) == 2:
        overall, checks = res
        try:
            overall = bool(overall) if overall is not None else None
        except Exception:
            overall = None
        checks = {str(k): bool(v) for k, v in (checks or {}).items()} if isinstance(checks, dict) else {}
        return overall, checks

    # dict shapes
    if isinstance(res, dict):
        overall = res.get('overall', None)
        try:
            overall = bool(overall) if overall is not None else None
        except Exception:
            overall = None

        # stages
        stages = res.get('stages')
        if isinstance(stages, dict):
            flat = {}
            for stage, checks in stages.items():
                if isinstance(checks, dict):
                    for k, v in checks.items():
                        flat[f"{stage}.{k}"] = bool(v)
            return overall, flat

        # explicit checks
        if isinstance(res.get('checks'), dict):
            checks_raw = res['checks']
        else:
            # flat dict of indicators
            checks_raw = {k: v for k, v in res.items() if k != 'overall' and not isinstance(v, (dict, list))}
        checks = {str(k): bool(v) for k, v in (checks_raw or {}).items()}
        return overall, checks

    # Fallback
    try:
        return bool(res), {}
    except Exception:
        return None, {}

# --- Minimal indicators enrichment before detection ---
def process_signals_and_maybe_enter(df, capital_data, last_ts: str, last_entry_by_pattern: dict, live=False, api=None, symbol=SYMBOL):
    """
    Check the latest candle, open every confirmed pattern, and return new trades.
    last_entry_by_pattern prevents opening the same pattern twice on the same minute.
    """
    if df is None or len(df) < 2:
        print("WARN DataFrame is too small to detect signals.")
        return []

    df = ensure_min_indicators(df)
    try:
        detectors = detect_Graphical_patterns(df)
    except Exception as exc:
        log_exception("Pattern detection failed; skipping this tick.", exc, key="detect_patterns")
        return []
    if not isinstance(detectors, dict):
        log_alert("ERROR", "Pattern detectors returned invalid output; skipping.", key="detect_patterns_type")
        return []
    last_idx = len(df) - 1
    last_close = _safe_float(df['close'].iloc[-1])
    if last_close is None or last_close <= 0:
        log_alert("WARN", "Last close price invalid; skipping.", key="price_invalid")
        return []
    new_trades: list[dict] = []

    # Precompute regime filters
    ema200 = df.get('ema_200')
    ema50 = df.get('ema_50')
    atr14 = df.get('atr_14')
    ema200_val = float(ema200.iloc[-1]) if isinstance(ema200, pd.Series) else None
    ema50_val = float(ema50.iloc[-1]) if isinstance(ema50, pd.Series) else None
    vol_ratio = None
    try:
        vol_ratio = float(atr14.iloc[-1] / last_close) if isinstance(atr14, pd.Series) else None
    except Exception:
        vol_ratio = None
    slope_sign = _slope_sign(ema200, window=EMA_SLOPE_WINDOW) if isinstance(ema200, pd.Series) else 0
    spread_ratio = None
    try:
        if ema50_val is not None and ema200_val is not None and last_close > 0:
            spread_ratio = abs(ema50_val - ema200_val) / last_close
    except Exception:
        spread_ratio = None

    for item in pattern_list:
        name = item.get("name")
        series = detectors.get(name)
        if series is None:
            continue

        try:
            last_val = bool(series.iloc[-1])
        except Exception:
            last_val = bool(series[-1]) if isinstance(series, (list, np.ndarray)) else False

        if not last_val:
            continue

        side = get_pattern_side(name)
        # Do not open the same pattern twice on the same minute.
        if last_entry_by_pattern.get(name) == last_ts:
            continue

        # Optional filters, disabled by default.
        if ENABLE_TREND_FILTERS:
            try:
                if ema200_val is not None:
                    slope_sym = 'up' if slope_sign > 0 else ('down' if slope_sign < 0 else '->')
                    band = float(TREND_BAND_PCT)
                    if side == 'LONG':
                        long_block = (last_close <= ema200_val * (1 - band)) and (slope_sign < 0)
                        if long_block:
                            print(f"Skip ({side}) '{name}': Against trend | Price {last_close:.2f} <= EMA200 {ema200_val:.2f} and slope {slope_sym}")
                            continue
                    else:
                        short_block = (last_close >= ema200_val * (1 + band)) and (slope_sign > 0)
                        if short_block:
                            print(f"Skip ({side}) '{name}': Against trend | Price {last_close:.2f} >= EMA200 {ema200_val:.2f} and slope {slope_sym}")
                            continue
            except Exception:
                pass

        if ENABLE_VOL_FILTER and vol_ratio is not None:
            if not (VOL_MIN <= vol_ratio <= VOL_MAX):
                continue

        if ENABLE_SPREAD_FILTER and spread_ratio is not None:
            if not (SPREAD_MIN <= spread_ratio <= SPREAD_MAX):
                continue

        overall, checks = normalize_confirmation_result(name, df, last_idx)
        outcomes = build_indicator_outcomes(name, checks if checks else (overall if overall is not None else False))
        valid_checks = sum(1 for v in outcomes.values() if bool(v))
        total_checks = len(outcomes)

        min_required = MIN_TRUE_INDICATORS_SHORT if side == "SHORT" else MIN_TRUE_INDICATORS_LONG
        if total_checks and valid_checks < min_required:
            print(
                f"Skip ({side}) '{name}': only {valid_checks}/{total_checks} indicator checks passed "
                f"(requires {min_required})."
            )
            continue

        # Reinforcement score is updated for analysis; entry gating is based on
        # the minimum indicator-check count above.
        try:
            key = ensure_reinforcement_key(name, outcomes)
        except Exception as exc:
            log_alert("WARN", f"Failed to build reinforcement key for {name}: {exc}", key="reinforcement_key")
            key = name
        scores = rm_load_scores()
        rec = scores.get(key, {}) if isinstance(scores, dict) else {}
        wins = int(rec.get("wins", 0)) if isinstance(rec, dict) else 0
        losses = int(rec.get("losses", 0)) if isinstance(rec, dict) else 0
        score = float(rec.get("score", get_score(key))) if isinstance(rec, dict) else 0.0

        # Simple risk scaling
        risk_pct = RISK_PCT_HIGH if total_checks and valid_checks == total_checks else RISK_PCT_LOW

        # ATR-based SL override (optional)
        sl_override_frac = None
        try:
            if vol_ratio is not None:
                sl_override_frac = max(0.004, min(0.02, 0.7 * vol_ratio))
        except Exception:
            sl_override_frac = None

        # Build a concise entry note with indicator outcomes and RL score
        checks_txt = ", ".join(f"{k}:{'PASS' if v else 'FAIL'}" for k, v in (outcomes or {}).items())
        checks_part = f"checks {valid_checks}/{total_checks} [{checks_txt}]" if total_checks else ""
        score_part = f"score {score:.1f} ({wins}W/{losses}L)"
        entry_note = " | ".join(part for part in [checks_part, score_part] if part)

        trade = enter_trade(
            df=df,
            breakout_idx=last_idx,
            pattern_type=name,
            capital_data={
                **(capital_data or {}),
                "risk_pct": risk_pct,
                "sl_override_frac": sl_override_frac,
            },
            live=live,
            api=api,
            symbol=symbol,
            context_key=key,
            entry_note=entry_note,
        )
        if trade:
            last_entry_by_pattern[name] = last_ts
            new_trades.append(trade)

    if not new_trades:
        print("Time No confirmed signal on the last candle.")
    return new_trades

#****************************************************************************************************************
#****************************************************************************************************************
# Step 5: Continuous runner (loops, CSV first, API fallback)

if __name__ == "__main__":
    try:
        initialize_all_files()

        # Startup prints the active source, budget, fees, and pattern count so a
        # reviewer can confirm the public reduced version is running.
        print("\nBot started (continuous mode)")
        print(f"Data source: {'CSV' if os.path.exists(CSV_PATH) else 'CSV missing - run generate_shared_candles_stocks.py'}")
        print(f"Capital per trade: {CAPITAL_PER_TRADE:,.0f}$ (no cap on concurrent trades)")
        print(f"Fees: buy {FEE_BUY*100:.2f}% | sell {FEE_SELL*100:.2f}%")
        print(f"Patterns loaded: {len(pattern_list)} (LONG:{sum(1 for p in pattern_list if p['type']=='LONG')} | SHORT:{sum(1 for p in pattern_list if p['type']=='SHORT')})\n")
        print(f"Files -> capital {CAPITAL_PATH}, reinforcement {REINFORCEMENT_PATH}, open {PENDING_PATTERN_PATH}")
        # print(f"Files -> capital {CAPITAL_PATH}, reinforcement {REINFORCEMENT_PATH}, open {PENDING_PATTERN_PATH}, closed {CLOSED_LOG_PATH}")

        day_high_txt = "N/A"
        day_low_txt = "N/A"
        last_change_txt = "N/A"
        initial_df = load_candles_from_csv(CSV_PATH, limit=390) if os.path.exists(CSV_PATH) else load_candles_from_api(limit=390)
        if initial_df is not None and len(initial_df) > 0:
            try:
                df_stats = initial_df.copy()
                df_stats['timestamp'] = pd.to_datetime(df_stats['timestamp'], errors='coerce')
                df_stats = df_stats.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                if not df_stats.empty:
                    last_ts = df_stats['timestamp'].iloc[-1]
                    day_start = last_ts.normalize()
                    day_slice = df_stats[df_stats['timestamp'] >= day_start]
                    if day_slice.empty:
                        day_slice = df_stats
                    start_price = float(day_slice['open'].iloc[0])
                    if start_price > 0:
                        day_high = float(day_slice['high'].max())
                        day_low = float(day_slice['low'].min())
                        last_close = float(day_slice['close'].iloc[-1])
                        day_high_txt = format_percent((day_high / start_price) - 1.0)
                        day_low_txt = format_percent((day_low / start_price) - 1.0)
                        last_change_txt = format_percent((last_close / start_price) - 1.0)
            except Exception as e:
                print(f"WARN Failed to compute intraday stats: {e}")

        print(f"{SYMBOL}: day high {day_high_txt} | day low {day_low_txt} | last change {last_change_txt}\n")

        # Wiring consistency check: every public pattern should have a detector,
        # a confirmer, and an indicator-rule schema.
        try:
            missing_confirm = []
            missing_rules = []
            for p in pattern_list:
                name = p.get('name')
                if name not in confirmation_functions:
                    missing_confirm.append(name)
                if name not in pattern_indicator_rules:
                    missing_rules.append(name)
            if missing_confirm:
                print(f"WARN Missing confirmation functions for: {', '.join(missing_confirm)}")
            if missing_rules:
                print(f"WARN Missing indicator rules for: {', '.join(missing_rules)}")
        except Exception as e:
            print(f"WARN Wiring check failed: {e}")

        # Restore open trades list (if any)
        open_trades = []
        pending_list = load_pending_pattern()
        if pending_list:
            print(f"Restoring {len(pending_list)} open trade(s) from {PENDING_PATTERN_PATH}")
            for p in pending_list:
                try:
                    if not isinstance(p, dict):
                        continue
                    entry_price = _safe_float(p.get("entry_price"))
                    if entry_price is None or entry_price <= 0:
                        log_alert("WARN", "Pending trade has invalid entry_price; skipping restore.", key="pending_restore")
                        continue
                    qty_val = _safe_float(p.get("quantity", 1.0), 1.0) or 1.0
                    open_trades.append({
                        "symbol": p.get("symbol", SYMBOL),
                        "signal": p.get("side", p.get("signal", "LONG")),
                        "pattern": p.get("pattern"),
                        "entry_time": p.get("timestamp", p.get("entry_time")),
                        "entry_price": entry_price,
                        "quantity": qty_val,
                        "status": p.get("status", "open"),
                        "current_level": int(p.get("current_level", 0)),
                        "tp_price": p.get("tp_price"),
                        "sl_price": p.get("sl_price"),
                        "context_key": p.get("context_key", p.get("pattern")),
                    })
                except Exception:
                    continue

        # Track last time per pattern to avoid duplicate open on the same candle
        last_entry_by_pattern: dict[str, str] = {}
        for t in open_trades:
            if t.get("pattern") and t.get("entry_time"):
                last_entry_by_pattern[t["pattern"]] = str(t["entry_time"])

        # Main loop. It sleeps outside the market window and wakes at the next
        # regular US open; inside the window it acts only on new candles.
        last_seen_ts = None
        same_candle_streak = 0
        no_candle_warned = False
        loop_error_streak = 0
        while True:
            try:
                session = market_session_status()
                if not session.get("is_active_window"):
                    sleep_until_next_market_open(session)
                    last_seen_ts = None
                    same_candle_streak = 0
                    no_candle_warned = False
                    continue

                # Load candles
                limit = 390
                if os.path.exists(CSV_PATH):
                    df = load_candles_from_csv(CSV_PATH, limit=limit)
                    source = "CSV"
                else:
                    df = load_candles_from_api(limit=limit)
                    source = "API"

                if df is None or len(df) < 2:
                    log_alert(
                        "WARN",
                        f"Not enough candles from {source}. Retrying in 10s...",
                        key="not_enough_candles",
                    )
                    time.sleep(10)
                    continue

                try:
                    df = filter_regular_market_hours(df)
                except Exception as e:
                    log_alert("WARN", f"Failed to filter regular hours: {e}", key="filter_hours")

                if FILL_MISSING_CANDLES:
                    df = fill_small_candle_gaps(
                        df,
                        source=source,
                        interval_minutes=INTERVAL_MINUTES,
                    )
                    if df is not None:
                        df = df.tail(limit).reset_index(drop=True)

                if df is None or len(df) < 2:
                    log_alert(
                        "INFO",
                        "Waiting for regular US market session data (market likely closed). Sleeping 60s...",
                        key="market_closed_wait",
                        throttle_seconds=300,
                    )
                    time.sleep(60)
                    continue

                # ISO timestamps for logging
                ts_series = None
                if 'timestamp' in df.columns:
                    try:
                        ts_series = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                        df['timestamp'] = ts_series.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                    except Exception:
                        ts_series = None

                if ts_series is None or ts_series.isna().all():
                    log_alert("WARN", "Timestamp parsing failed; retrying in 15s...", key="timestamp_parse")
                    time.sleep(15)
                    continue

                last_ts_dt = ts_series.iloc[-1]
                if pd.isna(last_ts_dt):
                    log_alert("WARN", "Latest timestamp is invalid; retrying.", key="timestamp_invalid")
                    time.sleep(10)
                    continue
                if last_ts_dt.tzinfo is None:
                    last_ts_dt = last_ts_dt.replace(tzinfo=timezone.utc)
                last_ts = df['timestamp'].iloc[-1]
                last_px = _safe_float(df['close'].iloc[-1])
                if last_px is None or last_px <= 0:
                    log_alert("WARN", "Last close price invalid; retrying.", key="price_invalid")
                    time.sleep(10)
                    continue

                data_delay_min = compute_data_delay_minutes(last_ts_dt, session.get("now_utc"))
                data_stale = False
                if data_delay_min is not None and data_delay_min > DATA_STALE_MINUTES:
                    data_stale = True
                    log_alert(
                        "WARN",
                        f"Data is stale (~{data_delay_min:.1f}m). Waiting for fresher bars...",
                        key="data_stale",
                    )
                    time.sleep(30)
                    continue

                unfilled_gap_report = detect_unfilled_candle_gap_issue(
                    df,
                    interval_minutes=INTERVAL_MINUTES,
                )
                unfilled_gap_issue = bool(unfilled_gap_report.get("block_entries"))
                if unfilled_gap_issue:
                    log_alert(
                        "WARN",
                        (
                            "Entry gate blocked: "
                            f"{unfilled_gap_report.get('missing_total', 0)} unfilled candle(s) "
                            f"in current {source} session."
                        ),
                        key=f"entry_block_missing:{source}",
                        details=unfilled_gap_report,
                        throttle_seconds=300,
                    )

                gap_min = None
                if len(ts_series) >= 2 and not pd.isna(ts_series.iloc[-2]):
                    try:
                        gap_min = (last_ts_dt - ts_series.iloc[-2]).total_seconds() / 60.0
                    except Exception:
                        gap_min = None
                expected_gap = max(1.0, float(INTERVAL_MINUTES))
                data_gap_issue = (
                    gap_min is not None
                    and gap_min > (expected_gap * DATA_GAP_MULTIPLIER)
                )
                if data_gap_issue:
                    log_alert(
                        "WARN",
                        f"Large candle gap detected (~{gap_min:.1f}m vs {expected_gap:.1f}m).",
                        key="candle_gap",
                    )

                new_candle = last_seen_ts != last_ts
                if new_candle:
                    same_candle_streak = 0
                    no_candle_warned = False
                    last_seen_ts = last_ts
                else:
                    same_candle_streak += 1
                    poll_seconds = max(1, int(POLL_INTERVAL_SECONDS))
                    expected_gap_seconds = max(1.0, float(INTERVAL_MINUTES)) * 60.0
                    warn_after_seconds = expected_gap_seconds * NO_CANDLE_WARN_MULTIPLIER
                    auto_warn_streak = int((warn_after_seconds + poll_seconds - 1) // poll_seconds)
                    warn_streak = max(MAX_SAME_CANDLE_STREAK, auto_warn_streak)
                    if same_candle_streak >= warn_streak and not no_candle_warned:
                        no_candle_warned = True
                        elapsed_seconds = same_candle_streak * poll_seconds
                        log_alert(
                            "WARN",
                            f"No new candle for ~{elapsed_seconds:.0f}s (poll {same_candle_streak}).",
                            key="no_new_candle",
                        )

                minutes_to_close = session.get("minutes_to_close")
                market_open_now = bool(session.get("is_open"))
                data_ok_for_entries = not data_gap_issue and not data_stale and not unfilled_gap_issue
                allow_new_entries = market_open_now and data_ok_for_entries and not (
                    minutes_to_close is not None and minutes_to_close <= ENTRY_BLOCK_BUFFER_MIN
                )

                if new_candle:
                    # Force-close any open trades shortly before market close
                    if open_trades and minutes_to_close is not None and minutes_to_close <= FORCED_EXIT_BUFFER_MIN:
                        print(f"Market closes in {minutes_to_close:.1f}m -> forcing exit of {len(open_trades)} open trade(s).")
                        for t in open_trades:
                            exit_trade(
                                t,
                                exit_price=last_px,
                                timestamp=last_ts,
                                symbol=t.get("symbol", SYMBOL),
                                live=False
                            )
                        open_trades = []
                        persist_open_trades(open_trades)

                    # Detect and open as many patterns as appear (no blocking)
                    new_trades: list[dict] = []
                    if allow_new_entries:
                        capital_data = {"capital": CAPITAL_PER_TRADE, "allocation_frac": 1.0}
                        new_trades = process_signals_and_maybe_enter(
                            df,
                            capital_data=capital_data,
                            last_ts=last_ts,
                            last_entry_by_pattern=last_entry_by_pattern,
                            live=False,
                            api=None,
                            symbol=SYMBOL
                        )
                        if new_trades:
                            open_trades.extend(new_trades)
                            persist_open_trades(open_trades)

                    # Update all open trades and close if SL hit
                    updated_trades = []
                    for t in open_trades:
                        updated, closed = on_tick_update_trade(
                            t,
                            current_price=last_px,
                            timestamp=last_ts,
                            symbol=t.get("symbol", SYMBOL),
                            live=False
                        )
                        if closed:
                            continue
                        updated_trades.append(updated)
                    open_trades = updated_trades
                    persist_open_trades(open_trades)

                    # Heartbeat status line each loop
                    print_heartbeat(
                        last_px,
                        open_trades,
                        new_trades=new_trades,
                        session=session,
                        data_delay_min=data_delay_min,
                        allow_new_entries=allow_new_entries,
                    )
                    if not market_open_now:
                        print("US market is closed. Skipping entries until the next session window.")
                    elif not allow_new_entries:
                        if not data_ok_for_entries:
                            print("Entry gate closed due to data gap/stale/unfilled candles.")
                        else:
                            print(f"Entry gate closed before market close (>={ENTRY_BLOCK_BUFFER_MIN}m buffer).")
                # Poll frequently for new candles
                try:
                    sleep_seconds = (
                        POST_CLOSE_POLL_SECONDS
                        if session.get("is_post_close_grace") and not market_open_now
                        else POLL_INTERVAL_SECONDS
                    )
                    time.sleep(max(1, int(sleep_seconds)))
                except Exception:
                    time.sleep(5)

                loop_error_streak = 0
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                loop_error_streak += 1
                log_exception("Loop error; retrying.", exc, key="loop_error")
                backoff = 5 if loop_error_streak < MAX_LOOP_ERROR_STREAK else 30
                time.sleep(backoff)
                continue

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"ERROR Fatal error: {e}")
