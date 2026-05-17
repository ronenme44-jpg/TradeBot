# tradeBot_main_7.py
#Bot71
# source ~/my_env/bin/activate
# python /Users/ronen/my_env/bot-trad/tradeBot_main_7.py
# sleep $(($(date -jf "%H:%M:%S" "16:29:00" +%s) - $(date +%s))) && python /Users/ronen/my_env/bot-trad/tradeBot_main_7.py
# ps aux | grep python

# לא נכנס לטרייד אם לא נכנסנו לפוזיציה טובה יותר ב 30 דקות קודם
# יש יותר מ 70 אחוזי הצלחה לתבנית וגם לפחות 2 טריידים וגם ציון מעל 250

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
from ttp_adapter import normalize_symbol, floor_shares, load_ttp_jsonl_to_df

import reinforcement_manager
# === Global paths ===
BASE_DIR = Path(__file__).resolve().parent


def _res(name: str) -> str:
    return str(BASE_DIR / name)


CSV_PATH = _res("shared_candles.csv")
TTP_FEED_JSONL = _res("shared_ttp_feed.jsonl")
CAPITAL_PATH = _res("Bot71_capital.json")
REINFORCEMENT_PATH = _res("Bot71_reinforcement_scores.json")
PENDING_PATTERN_PATH = _res("Bot71_pending_pattern.json")
CLOSED_LOG_PATH = _res("Bot71_closed_trades_log.json")
SYMBOL = normalize_symbol("AAPL") or "AAPL"  # default equity symbol
BOT_ID = "Bot71"

DEFAULT_CAPITAL = 40000
DEFAULT_CAPITAL_PAYLOAD = {"capital": DEFAULT_CAPITAL}
DEFAULT_REINFORCEMENT_PAYLOAD = {}
DEFAULT_PENDING_PATTERN = None
DEFAULT_CLOSED_LOG = []

# --- Resilience / alerting config ---
ALERT_LOG_PATH = _res(f"{BOT_ID}_alerts.jsonl")
ALERT_THROTTLE_SECONDS = 120
DATA_STALE_MINUTES = 20
FREE_DATA_DELAY_MINUTES = DATA_STALE_MINUTES
DATA_GAP_MULTIPLIER = 3.0
MAX_SAME_CANDLE_STREAK = 3
NO_CANDLE_WARN_MULTIPLIER = 2.0
FILL_MISSING_CANDLES = True
MAX_MISSING_CANDLES_PER_GAP = 2
MAX_MISSING_CANDLES_PER_SESSION = 3
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

    prefix = {"ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️"}.get(level_norm, "ℹ️")
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


# --- Better-price blocked opportunity tracking ---
_blocked_opportunities: list[dict] = []


def _parse_timestamp_utc(value) -> datetime | None:
    """Parse timestamps to UTC-aware datetime for comparisons."""
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    if isinstance(ts, pd.Series):
        try:
            ts = ts.iloc[-1]
        except Exception:
            return None
        if ts is None or pd.isna(ts):
            return None
    try:
        return ts.to_pydatetime()
    except Exception:
        if isinstance(ts, datetime):
            return ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return None


def _prune_blocked_opportunities(now_ts: datetime) -> None:
    """Keep only blocked opportunities within the recent lookback window."""
    if not isinstance(now_ts, datetime):
        return
    cutoff = now_ts - timedelta(minutes=BLOCKED_BETTER_PRICE_LOOKBACK_MIN)
    kept: list[dict] = []
    for item in _blocked_opportunities:
        ts = item.get("timestamp")
        if isinstance(ts, datetime) and ts >= cutoff:
            kept.append(item)
    _blocked_opportunities[:] = kept


def record_blocked_opportunity(
    side: str,
    price: float,
    timestamp,
    reason: str,
    *,
    pattern: str | None = None,
    key: str | None = None,
) -> None:
    """Store a blocked opportunity for later 'better price' checks."""
    side_norm = str(side or "").upper()
    if side_norm not in ("LONG", "SHORT"):
        return
    px = _safe_float(price)
    ts = _parse_timestamp_utc(timestamp)
    if px is None or px <= 0 or ts is None:
        return

    _prune_blocked_opportunities(ts)
    if _blocked_opportunities:
        last = _blocked_opportunities[-1]
        if (
            last.get("side") == side_norm
            and last.get("price") == px
            and last.get("timestamp") == ts
            and last.get("reason") == reason
        ):
            return

    _blocked_opportunities.append(
        {
            "timestamp": ts,
            "side": side_norm,
            "price": px,
            "reason": str(reason or "unknown"),
            "pattern": pattern,
            "key": key,
        }
    )
    if len(_blocked_opportunities) > MAX_BLOCKED_OPPORTUNITIES:
        _blocked_opportunities[:] = _blocked_opportunities[-MAX_BLOCKED_OPPORTUNITIES:]


def find_recent_better_block(
    side: str,
    current_price: float,
    now_ts,
) -> dict | None:
    """Return the best blocked opportunity (if any) that was better-priced within the lookback."""
    side_norm = str(side or "").upper()
    if side_norm not in ("LONG", "SHORT"):
        return None
    px = _safe_float(current_price)
    ts = _parse_timestamp_utc(now_ts)
    if px is None or px <= 0 or ts is None:
        return None

    _prune_blocked_opportunities(ts)
    best = None
    for item in _blocked_opportunities:
        if item.get("side") != side_norm:
            continue
        its = item.get("timestamp")
        ipx = item.get("price")
        if not isinstance(its, datetime) or ipx is None:
            continue
        if its > ts:
            continue
        if side_norm == "LONG":
            if ipx < px and (best is None or ipx < best.get("price", ipx)):
                best = item
        else:
            if ipx > px and (best is None or ipx > best.get("price", ipx)):
                best = item
    if best is None:
        return None
    age_min = max(0.0, (ts - best["timestamp"]).total_seconds() / 60.0)
    return {**best, "age_min": age_min}

reinforcement_manager.REINFORCEMENT_PATH = REINFORCEMENT_PATH
# Alias to avoid NameError in any scope that refers to load_scores
load_scores = reinforcement_manager.load_scores

# --- Project modules ---
from tradeBot_Thresholds import THRESHOLDS
from tradeBot_Graphical_Pattern import detect_Graphical_patterns
from tradeBot_indicators_pattern import confirmation_functions
from reinforcement_manager import update_score_on_close, load_scores as rm_load_scores
from tradeBot_Indicator_Rules import pattern_indicator_rules, build_pattern_key
from terminal_display import render_status as render_snapshot

# === Global parameters ===
TP_BASE = 0.01    # 🎯 יעד רווח ראשוני: 1%
SL_BASE = -0.005  # 🛑 סטופ לוס ראשוני: -0.4%
INTERVAL_MINUTES = 1  # ⏱️ טווח זמן לנרות (1 דקות)

# --- Side balancing guard (reduce short over-entry) ---
# Require a higher minimum number of passing indicators for SHORT vs LONG.
# This corrects a tendency to over-trigger shorts without rewriting all detectors.
MIN_TRUE_INDICATORS_LONG = 2
MIN_TRUE_INDICATORS_SHORT = 3

# Feature toggles for entry filters
ENABLE_TREND_FILTERS = False   # price vs EMA200 + slope (disabled per request)
ENABLE_VOL_FILTER    = False   # ATR/close band (disabled per request)
ENABLE_SPREAD_FILTER = False   # EMA50/200 spread band (disabled per request)

# Better-price guard: block entries if a better-priced blocked opportunity exists recently
BLOCKED_BETTER_PRICE_LOOKBACK_MIN = 30
MAX_BLOCKED_OPPORTUNITIES = 200

# Risk per trade (fraction of capital)
RISK_PCT_LOW  = 0.005  # 0.5%
RISK_PCT_HIGH = 0.010  # 1.0%
RISK_PCT_CAUTIOUS = 0.0025  # 0.25% exploration sizing

# Reinforcement gating for template keys
RL_BLOCK_SCORE = -6.0
MIN_ENTRY_SCORE = 500.0
MIN_ENTRY_TRADES = 3
MIN_ENTRY_WIN_RATE = 0.70

# Market regime filters
VOL_MIN = 0.003   # 0.30% ATR/close minimum
VOL_MAX = 0.020   # 2.00% ATR/close maximum
SPREAD_MIN = 0.002  # 0.20% |ema50-ema200|/close minimum
SPREAD_MAX = 0.030  # 3.00% maximum
EMA_SLOPE_WINDOW = 20  # bars to estimate ema200 slope sign
TREND_BAND_PCT = 0.0005  # 0.05% neutral band around EMA200 for trend filter

# Market session (US equities)
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
    """Return a New York timestamp on the same date at the requested minute of day."""
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
    """Format sleep duration for terminal logs."""
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
    """Keep only regular session candles (Mon–Fri, 09:30–16:00 NY time)."""
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
    Fill in-session gaps between consecutive candles using surrounding prices.
    Keeps the bot resilient when candles are missing inside the window.
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
    large_gap_sessions = 0
    unfilled_gaps = 0

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

        if missing_total > max_missing_per_session or any(m > max_missing_per_gap for _, m in gaps):
            large_gap_sessions += 1

        filled_rows = []
        for idx, missing in gaps:
            prev_close = _safe_float(group.at[idx, "close"])
            if prev_close is None or prev_close <= 0:
                prev_close = _safe_float(group.at[idx, "open"])
            next_anchor = _safe_float(group.at[idx + 1, "open"])
            if next_anchor is None or next_anchor <= 0:
                next_anchor = _safe_float(group.at[idx + 1, "close"])
            if prev_close is None or prev_close <= 0 or next_anchor is None or next_anchor <= 0:
                unfilled_gaps += 1
                continue
            base_ts = group.at[idx, "timestamp"]
            prev_step_close = prev_close
            for step in range(1, missing + 1):
                ratio = step / (missing + 1)
                close_px = prev_close + ((next_anchor - prev_close) * ratio)
                open_px = prev_step_close
                filled_rows.append({
                    "timestamp": base_ts + expected_gap * step,
                    "open": open_px,
                    "high": max(open_px, close_px),
                    "low": min(open_px, close_px),
                    "close": close_px,
                    "volume": 0.0,
                })
                prev_step_close = close_px

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
                "large_gap_sessions": large_gap_sessions,
                "unfilled_gaps": unfilled_gaps,
                "method": "linear_interpolate_prev_close_to_next_open",
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
                "large_gap_sessions": large_gap_sessions,
                "unfilled_gaps": unfilled_gaps,
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
    print("────────────────────────────────────────────────────────")
    arrow = "⬆️" if side == "LONG" else "⬇️"
    outcome = "✅ success" if net_usd >= 0 else "❌ fail"
    print(f"💼 Trade Summary {arrow} {side}")
    print(f"Qty: {int(qty)} | Levels: {level}")
    print(f"Entry: {entry:.2f} @ {entry_ts}")
    print(f"Exit : {exit_price:.2f} @ {exit_ts} (reason: SL hit)")
    print(f"Gross: {format_percent(gross_frac)} ({_fmt_usd(gross_usd)})")
    print(f"Fees : {_fmt_usd(-buy_fee_usd)} entry + {_fmt_usd(-sell_fee_usd)} exit = {_fmt_usd(-total_fees_usd)} total")
    print(f"Net  : {format_percent(net_frac)} ({_fmt_usd(net_usd)})  → {outcome}")
    print("────────────────────────────────────────────────────────")

# לתצוגה יפה ומסודרת
pattern_list = [
    {"name": "bullish_engulfing", "type": "LONG"},
    {"name": "hammer", "type": "LONG"},
    {"name": "morning_star", "type": "LONG"},
    {"name": "piercing_pattern", "type": "LONG"},
    {"name": "tweezer_bottom", "type": "LONG"},
    {"name": "bullish_harami", "type": "LONG"},
    {"name": "inverted_hammer", "type": "LONG"},
    {"name": "three_white_soldiers_simple", "type": "LONG"},
    {"name": "rising_three_methods", "type": "LONG"},
    {"name": "bullish_marubozu", "type": "LONG"},
    {"name": "three_inside_up", "type": "LONG"},
    {"name": "bullish_kicker", "type": "LONG"},
    {"name": "bull_flag", "type": "LONG"},
    {"name": "flat_top_breakout", "type": "LONG"},
    {"name": "cup_and_handle", "type": "LONG"},
    {"name": "ascending_triangle", "type": "LONG"},
    {"name": "rounding_bottom", "type": "LONG"},
    {"name": "golden_cross", "type": "LONG"},
    {"name": "ema_channel_up", "type": "LONG"},
    {"name": "higher_highs_lows", "type": "LONG"},
    {"name": "ema_crossover_pullback", "type": "LONG"},
    {"name": "ascending_channel_breakout", "type": "LONG"},
    {"name": "trend_reversal_up", "type": "LONG"},
    {"name": "double_bottom_breakout", "type": "LONG"},
    {"name": "falling_wedge_breakout", "type": "LONG"},
    {"name": "volume_dryup_spike", "type": "LONG"},
    {"name": "ema_squeeze_breakout", "type": "LONG"},
    {"name": "oversold_bounce_reversal", "type": "LONG"},
    {"name": "consolidation_breakout_up", "type": "LONG"},
    {"name": "bullish_abandoned_baby", "type": "LONG"},
    {"name": "bullish_breakaway_gap", "type": "LONG"},
    {"name": "bullish_railway_tracks", "type": "LONG"},
    {"name": "inside_bar_breakout_up", "type": "LONG"},
    {"name": "bullish_pennant_breakout", "type": "LONG"},
    {"name": "ema21_pullback_bounce", "type": "LONG"},
    {"name": "ema50_reclaim", "type": "LONG"},
    {"name": "higher_low_retest_breakout", "type": "LONG"},
    {"name": "volume_climax_reversal_up", "type": "LONG"},
    {"name": "triple_bottom_breakout", "type": "LONG"},
    {"name": "range_shift_up", "type": "LONG"},
    {"name": "ma_stack_trend_long", "type": "LONG"},
    {"name": "vcp_breakout_long", "type": "LONG"},
    {"name": "hourly_trend_continuation_breakout", "type": "LONG"},
    {"name": "hourly_ema50_pullback_bounce", "type": "LONG"},
    {"name": "hourly_flat_base_breakout", "type": "LONG"},
    {"name": "hourly_vol_squeeze_hour_breakout", "type": "LONG"},
    {"name": "hourly_higher_low_spring", "type": "LONG"},
    {"name": "hourly_volume_climb_breakout", "type": "LONG"},
    {"name": "hourly_mean_reversion_snap_up", "type": "LONG"},
    {"name": "hourly_open_drive_continuation", "type": "LONG"},
    {"name": "close_trend_breakout_10", "type": "LONG"},
    {"name": "close_trend_breakout_20", "type": "LONG"},
    {"name": "close_trend_breakout_30", "type": "LONG"},
    {"name": "close_flat_base_breakout_close", "type": "LONG"},
    {"name": "close_squeeze_breakout_close", "type": "LONG"},
    {"name": "close_pullback_reclaim", "type": "LONG"},
    {"name": "close_regression_breakout", "type": "LONG"},
    {"name": "close_v_reversal_up", "type": "LONG"},
    {"name": "close_higher_low_breakout", "type": "LONG"},
    {"name": "close_mean_reversion_snap_up", "type": "LONG"},
    {"name": "close_step_up_expansion", "type": "LONG"},
    {"name": "close_range_shift_up_close", "type": "LONG"},
    {"name": "close_momentum_breakout", "type": "LONG"},
    {"name": "close_rsi_dip_breakout", "type": "LONG"},
    {"name": "close_ma_stack_breakout", "type": "LONG"},
    {"name": "bearish_engulfing", "type": "SHORT"},
    {"name": "shooting_star", "type": "SHORT"},
    {"name": "evening_star", "type": "SHORT"},
    {"name": "three_black_crows", "type": "SHORT"},
    {"name": "dark_cloud_cover", "type": "SHORT"},
    {"name": "tweezer_top", "type": "SHORT"},
    {"name": "hanging_man", "type": "SHORT"},
    {"name": "bearish_harami", "type": "SHORT"},
    {"name": "falling_three_methods", "type": "SHORT"},
    {"name": "bearish_marubozu", "type": "SHORT"},
    {"name": "three_inside_down", "type": "SHORT"},
    {"name": "bearish_kicker", "type": "SHORT"},
    {"name": "descending_channel_breakdown", "type": "SHORT"},
    {"name": "lower_highs_lows", "type": "SHORT"},
    {"name": "ema_channel_down", "type": "SHORT"},
    {"name": "bearish_ema_cross", "type": "SHORT"},
    {"name": "double_top", "type": "SHORT"},
    {"name": "ema_crossover_down", "type": "SHORT"},
    {"name": "lower_high_breakdown", "type": "SHORT"},
    {"name": "trend_channel_down", "type": "SHORT"},
    {"name": "bear_flag", "type": "SHORT"},
    {"name": "shooting_star_breakdown", "type": "SHORT"},
    {"name": "double_top_breakdown", "type": "SHORT"},
    {"name": "bearish_ema_cross_drop", "type": "SHORT"},
    {"name": "falling_wedge_breakdown", "type": "SHORT"},
    {"name": "trend_exhaustion_breakdown", "type": "SHORT"},
    {"name": "volatility_squeeze_breakdown", "type": "SHORT"},
    {"name": "rsi_divergence_breakdown", "type": "SHORT"},
    {"name": "overbought_pullback_reversal", "type": "SHORT"},
    {"name": "bearish_abandoned_baby", "type": "SHORT"},
    {"name": "bearish_breakaway_gap", "type": "SHORT"},
    {"name": "bearish_railway_tracks", "type": "SHORT"},
    {"name": "inside_bar_breakdown", "type": "SHORT"},
    {"name": "bearish_pennant_breakdown", "type": "SHORT"},
    {"name": "ema21_pullback_reject", "type": "SHORT"},
    {"name": "ema50_failure", "type": "SHORT"},
    {"name": "lower_high_retest_breakdown", "type": "SHORT"},
    {"name": "volume_climax_reversal_down", "type": "SHORT"},
    {"name": "triple_top_breakdown", "type": "SHORT"},
    {"name": "range_shift_down", "type": "SHORT"},
    {"name": "ma_stack_trend_short", "type": "SHORT"},
    {"name": "vcp_breakdown_short", "type": "SHORT"},
    {"name": "hourly_trend_rollover_breakdown", "type": "SHORT"},
    {"name": "hourly_ema50_pullback_reject", "type": "SHORT"},
    {"name": "hourly_flat_base_breakdown", "type": "SHORT"},
    {"name": "hourly_volume_climax_drop", "type": "SHORT"},
    {"name": "hourly_lower_high_fade", "type": "SHORT"},
    {"name": "hourly_vol_expansion_breakdown", "type": "SHORT"},
    {"name": "hourly_mean_reversion_snap_down", "type": "SHORT"},
    {"name": "hourly_open_drive_fail", "type": "SHORT"},
    {"name": "close_trend_breakdown_10", "type": "SHORT"},
    {"name": "close_trend_breakdown_20", "type": "SHORT"},
    {"name": "close_trend_breakdown_30", "type": "SHORT"},
    {"name": "close_flat_base_breakdown_close", "type": "SHORT"},
    {"name": "close_squeeze_breakdown_close", "type": "SHORT"},
    {"name": "close_lower_high_reject", "type": "SHORT"},
    {"name": "close_regression_breakdown", "type": "SHORT"},
    {"name": "close_v_reversal_down", "type": "SHORT"},
    {"name": "close_lower_high_breakdown", "type": "SHORT"},
    {"name": "close_mean_reversion_snap_down", "type": "SHORT"},
    {"name": "close_step_down_expansion", "type": "SHORT"},
    {"name": "close_range_shift_down_close", "type": "SHORT"},
    {"name": "close_momentum_breakdown", "type": "SHORT"},
    {"name": "close_rsi_pop_breakdown", "type": "SHORT"},
    {"name": "close_ma_stack_breakdown", "type": "SHORT"},
    {"name": "bullish_engulfing_negative", "type": "SHORT"},
    {"name": "hammer_negative", "type": "SHORT"},
    {"name": "morning_star_negative", "type": "SHORT"},
    {"name": "piercing_pattern_negative", "type": "SHORT"},
    {"name": "tweezer_bottom_negative", "type": "SHORT"},
    {"name": "bullish_harami_negative", "type": "SHORT"},
    {"name": "inverted_hammer_negative", "type": "SHORT"},
    {"name": "three_white_soldiers_simple_negative", "type": "SHORT"},
    {"name": "rising_three_methods_negative", "type": "SHORT"},
    {"name": "bullish_marubozu_negative", "type": "SHORT"},
    {"name": "three_inside_up_negative", "type": "SHORT"},
    {"name": "bullish_kicker_negative", "type": "SHORT"},
    {"name": "bull_flag_negative", "type": "SHORT"},
    {"name": "flat_top_breakout_negative", "type": "SHORT"},
    {"name": "cup_and_handle_negative", "type": "SHORT"},
    {"name": "ascending_triangle_negative", "type": "SHORT"},
    {"name": "rounding_bottom_negative", "type": "SHORT"},
    {"name": "golden_cross_negative", "type": "SHORT"},
    {"name": "ema_channel_up_negative", "type": "SHORT"},
    {"name": "higher_highs_lows_negative", "type": "SHORT"},
    {"name": "ema_crossover_pullback_negative", "type": "SHORT"},
    {"name": "ascending_channel_breakout_negative", "type": "SHORT"},
    {"name": "trend_reversal_up_negative", "type": "SHORT"},
    {"name": "double_bottom_breakout_negative", "type": "SHORT"},
    {"name": "falling_wedge_breakout_negative", "type": "SHORT"},
    {"name": "volume_dryup_spike_negative", "type": "SHORT"},
    {"name": "ema_squeeze_breakout_negative", "type": "SHORT"},
    {"name": "oversold_bounce_reversal_negative", "type": "SHORT"},
    {"name": "consolidation_breakout_up_negative", "type": "SHORT"},
    {"name": "bullish_abandoned_baby_negative", "type": "SHORT"},
    {"name": "bullish_breakaway_gap_negative", "type": "SHORT"},
    {"name": "bullish_railway_tracks_negative", "type": "SHORT"},
    {"name": "inside_bar_breakout_up_negative", "type": "SHORT"},
    {"name": "bullish_pennant_breakout_negative", "type": "SHORT"},
    {"name": "ema21_pullback_bounce_negative", "type": "SHORT"},
    {"name": "ema50_reclaim_negative", "type": "SHORT"},
    {"name": "higher_low_retest_breakout_negative", "type": "SHORT"},
    {"name": "volume_climax_reversal_up_negative", "type": "SHORT"},
    {"name": "triple_bottom_breakout_negative", "type": "SHORT"},
    {"name": "range_shift_up_negative", "type": "SHORT"},
    {"name": "ma_stack_trend_long_negative", "type": "SHORT"},
    {"name": "vcp_breakout_long_negative", "type": "SHORT"},
    {"name": "hourly_trend_continuation_breakout_negative", "type": "SHORT"},
    {"name": "hourly_ema50_pullback_bounce_negative", "type": "SHORT"},
    {"name": "hourly_flat_base_breakout_negative", "type": "SHORT"},
    {"name": "hourly_vol_squeeze_hour_breakout_negative", "type": "SHORT"},
    {"name": "hourly_higher_low_spring_negative", "type": "SHORT"},
    {"name": "hourly_volume_climb_breakout_negative", "type": "SHORT"},
    {"name": "hourly_mean_reversion_snap_up_negative", "type": "SHORT"},
    {"name": "hourly_open_drive_continuation_negative", "type": "SHORT"},
    {"name": "close_trend_breakout_10_negative", "type": "SHORT"},
    {"name": "close_trend_breakout_20_negative", "type": "SHORT"},
    {"name": "close_trend_breakout_30_negative", "type": "SHORT"},
    {"name": "close_flat_base_breakout_close_negative", "type": "SHORT"},
    {"name": "close_squeeze_breakout_close_negative", "type": "SHORT"},
    {"name": "close_pullback_reclaim_negative", "type": "SHORT"},
    {"name": "close_regression_breakout_negative", "type": "SHORT"},
    {"name": "close_v_reversal_up_negative", "type": "SHORT"},
    {"name": "close_higher_low_breakout_negative", "type": "SHORT"},
    {"name": "close_mean_reversion_snap_up_negative", "type": "SHORT"},
    {"name": "close_step_up_expansion_negative", "type": "SHORT"},
    {"name": "close_range_shift_up_close_negative", "type": "SHORT"},
    {"name": "close_momentum_breakout_negative", "type": "SHORT"},
    {"name": "close_rsi_dip_breakout_negative", "type": "SHORT"},
    {"name": "close_ma_stack_breakout_negative", "type": "SHORT"},
    {"name": "bearish_engulfing_negative", "type": "LONG"},
    {"name": "shooting_star_negative", "type": "LONG"},
    {"name": "evening_star_negative", "type": "LONG"},
    {"name": "three_black_crows_negative", "type": "LONG"},
    {"name": "dark_cloud_cover_negative", "type": "LONG"},
    {"name": "tweezer_top_negative", "type": "LONG"},
    {"name": "hanging_man_negative", "type": "LONG"},
    {"name": "bearish_harami_negative", "type": "LONG"},
    {"name": "falling_three_methods_negative", "type": "LONG"},
    {"name": "bearish_marubozu_negative", "type": "LONG"},
    {"name": "three_inside_down_negative", "type": "LONG"},
    {"name": "bearish_kicker_negative", "type": "LONG"},
    {"name": "descending_channel_breakdown_negative", "type": "LONG"},
    {"name": "lower_highs_lows_negative", "type": "LONG"},
    {"name": "ema_channel_down_negative", "type": "LONG"},
    {"name": "bearish_ema_cross_negative", "type": "LONG"},
    {"name": "double_top_negative", "type": "LONG"},
    {"name": "ema_crossover_down_negative", "type": "LONG"},
    {"name": "lower_high_breakdown_negative", "type": "LONG"},
    {"name": "trend_channel_down_negative", "type": "LONG"},
    {"name": "bear_flag_negative", "type": "LONG"},
    {"name": "shooting_star_breakdown_negative", "type": "LONG"},
    {"name": "double_top_breakdown_negative", "type": "LONG"},
    {"name": "bearish_ema_cross_drop_negative", "type": "LONG"},
    {"name": "falling_wedge_breakdown_negative", "type": "LONG"},
    {"name": "trend_exhaustion_breakdown_negative", "type": "LONG"},
    {"name": "volatility_squeeze_breakdown_negative", "type": "LONG"},
    {"name": "rsi_divergence_breakdown_negative", "type": "LONG"},
    {"name": "overbought_pullback_reversal_negative", "type": "LONG"},
    {"name": "bearish_abandoned_baby_negative", "type": "LONG"},
    {"name": "bearish_breakaway_gap_negative", "type": "LONG"},
    {"name": "bearish_railway_tracks_negative", "type": "LONG"},
    {"name": "inside_bar_breakdown_negative", "type": "LONG"},
    {"name": "bearish_pennant_breakdown_negative", "type": "LONG"},
    {"name": "ema21_pullback_reject_negative", "type": "LONG"},
    {"name": "ema50_failure_negative", "type": "LONG"},
    {"name": "lower_high_retest_breakdown_negative", "type": "LONG"},
    {"name": "volume_climax_reversal_down_negative", "type": "LONG"},
    {"name": "triple_top_breakdown_negative", "type": "LONG"},
    {"name": "range_shift_down_negative", "type": "LONG"},
    {"name": "ma_stack_trend_short_negative", "type": "LONG"},
    {"name": "vcp_breakdown_short_negative", "type": "LONG"},
    {"name": "hourly_trend_rollover_breakdown_negative", "type": "LONG"},
    {"name": "hourly_ema50_pullback_reject_negative", "type": "LONG"},
    {"name": "hourly_flat_base_breakdown_negative", "type": "LONG"},
    {"name": "hourly_volume_climax_drop_negative", "type": "LONG"},
    {"name": "hourly_lower_high_fade_negative", "type": "LONG"},
    {"name": "hourly_vol_expansion_breakdown_negative", "type": "LONG"},
    {"name": "hourly_mean_reversion_snap_down_negative", "type": "LONG"},
    {"name": "hourly_open_drive_fail_negative", "type": "LONG"},
    {"name": "close_trend_breakdown_10_negative", "type": "LONG"},
    {"name": "close_trend_breakdown_20_negative", "type": "LONG"},
    {"name": "close_trend_breakdown_30_negative", "type": "LONG"},
    {"name": "close_flat_base_breakdown_close_negative", "type": "LONG"},
    {"name": "close_squeeze_breakdown_close_negative", "type": "LONG"},
    {"name": "close_lower_high_reject_negative", "type": "LONG"},
    {"name": "close_regression_breakdown_negative", "type": "LONG"},
    {"name": "close_v_reversal_down_negative", "type": "LONG"},
    {"name": "close_lower_high_breakdown_negative", "type": "LONG"},
    {"name": "close_mean_reversion_snap_down_negative", "type": "LONG"},
    {"name": "close_step_down_expansion_negative", "type": "LONG"},
    {"name": "close_range_shift_down_close_negative", "type": "LONG"},
    {"name": "close_momentum_breakdown_negative", "type": "LONG"},
    {"name": "close_rsi_pop_breakdown_negative", "type": "LONG"},
    {"name": "close_ma_stack_breakdown_negative", "type": "LONG"},
    {"name": "bullish_engulfing_5Minutes", "type": "LONG"},
    {"name": "bullish_engulfing_5Minutes_negative", "type": "SHORT"},
    {"name": "bullish_engulfing_15Minutes", "type": "LONG"},
    {"name": "bullish_engulfing_15Minutes_negative", "type": "SHORT"},
    {"name": "hammer_5Minutes", "type": "LONG"},
    {"name": "hammer_5Minutes_negative", "type": "SHORT"},
    {"name": "hammer_15Minutes", "type": "LONG"},
    {"name": "hammer_15Minutes_negative", "type": "SHORT"},
    {"name": "morning_star_5Minutes", "type": "LONG"},
    {"name": "morning_star_5Minutes_negative", "type": "SHORT"},
    {"name": "morning_star_15Minutes", "type": "LONG"},
    {"name": "morning_star_15Minutes_negative", "type": "SHORT"},
    {"name": "piercing_pattern_5Minutes", "type": "LONG"},
    {"name": "piercing_pattern_5Minutes_negative", "type": "SHORT"},
    {"name": "piercing_pattern_15Minutes", "type": "LONG"},
    {"name": "piercing_pattern_15Minutes_negative", "type": "SHORT"},
    {"name": "tweezer_bottom_5Minutes", "type": "LONG"},
    {"name": "tweezer_bottom_5Minutes_negative", "type": "SHORT"},
    {"name": "tweezer_bottom_15Minutes", "type": "LONG"},
    {"name": "tweezer_bottom_15Minutes_negative", "type": "SHORT"},
    {"name": "bullish_harami_5Minutes", "type": "LONG"},
    {"name": "bullish_harami_5Minutes_negative", "type": "SHORT"},
    {"name": "bullish_harami_15Minutes", "type": "LONG"},
    {"name": "bullish_harami_15Minutes_negative", "type": "SHORT"},
    {"name": "inverted_hammer_5Minutes", "type": "LONG"},
    {"name": "inverted_hammer_5Minutes_negative", "type": "SHORT"},
    {"name": "inverted_hammer_15Minutes", "type": "LONG"},
    {"name": "inverted_hammer_15Minutes_negative", "type": "SHORT"},
    {"name": "three_white_soldiers_simple_5Minutes", "type": "LONG"},
    {"name": "three_white_soldiers_simple_5Minutes_negative", "type": "SHORT"},
    {"name": "three_white_soldiers_simple_15Minutes", "type": "LONG"},
    {"name": "three_white_soldiers_simple_15Minutes_negative", "type": "SHORT"},
    {"name": "rising_three_methods_5Minutes", "type": "LONG"},
    {"name": "rising_three_methods_5Minutes_negative", "type": "SHORT"},
    {"name": "rising_three_methods_15Minutes", "type": "LONG"},
    {"name": "rising_three_methods_15Minutes_negative", "type": "SHORT"},
    {"name": "bullish_marubozu_5Minutes", "type": "LONG"},
    {"name": "bullish_marubozu_5Minutes_negative", "type": "SHORT"},
    {"name": "bullish_marubozu_15Minutes", "type": "LONG"},
    {"name": "bullish_marubozu_15Minutes_negative", "type": "SHORT"},
    {"name": "three_inside_up_5Minutes", "type": "LONG"},
    {"name": "three_inside_up_5Minutes_negative", "type": "SHORT"},
    {"name": "three_inside_up_15Minutes", "type": "LONG"},
    {"name": "three_inside_up_15Minutes_negative", "type": "SHORT"},
    {"name": "bullish_kicker_5Minutes", "type": "LONG"},
    {"name": "bullish_kicker_5Minutes_negative", "type": "SHORT"},
    {"name": "bullish_kicker_15Minutes", "type": "LONG"},
    {"name": "bullish_kicker_15Minutes_negative", "type": "SHORT"},
    {"name": "bull_flag_5Minutes", "type": "LONG"},
    {"name": "bull_flag_5Minutes_negative", "type": "SHORT"},
    {"name": "bull_flag_15Minutes", "type": "LONG"},
    {"name": "bull_flag_15Minutes_negative", "type": "SHORT"},
    {"name": "flat_top_breakout_5Minutes", "type": "LONG"},
    {"name": "flat_top_breakout_5Minutes_negative", "type": "SHORT"},
    {"name": "flat_top_breakout_15Minutes", "type": "LONG"},
    {"name": "flat_top_breakout_15Minutes_negative", "type": "SHORT"},
    {"name": "cup_and_handle_5Minutes", "type": "LONG"},
    {"name": "cup_and_handle_5Minutes_negative", "type": "SHORT"},
    {"name": "cup_and_handle_15Minutes", "type": "LONG"},
    {"name": "cup_and_handle_15Minutes_negative", "type": "SHORT"},
    {"name": "ascending_triangle_5Minutes", "type": "LONG"},
    {"name": "ascending_triangle_5Minutes_negative", "type": "SHORT"},
    {"name": "ascending_triangle_15Minutes", "type": "LONG"},
    {"name": "ascending_triangle_15Minutes_negative", "type": "SHORT"},
    {"name": "rounding_bottom_5Minutes", "type": "LONG"},
    {"name": "rounding_bottom_5Minutes_negative", "type": "SHORT"},
    {"name": "rounding_bottom_15Minutes", "type": "LONG"},
    {"name": "rounding_bottom_15Minutes_negative", "type": "SHORT"},
    {"name": "golden_cross_5Minutes", "type": "LONG"},
    {"name": "golden_cross_5Minutes_negative", "type": "SHORT"},
    {"name": "golden_cross_15Minutes", "type": "LONG"},
    {"name": "golden_cross_15Minutes_negative", "type": "SHORT"},
    {"name": "ema_channel_up_5Minutes", "type": "LONG"},
    {"name": "ema_channel_up_5Minutes_negative", "type": "SHORT"},
    {"name": "ema_channel_up_15Minutes", "type": "LONG"},
    {"name": "ema_channel_up_15Minutes_negative", "type": "SHORT"},
    {"name": "higher_highs_lows_5Minutes", "type": "LONG"},
    {"name": "higher_highs_lows_5Minutes_negative", "type": "SHORT"},
    {"name": "higher_highs_lows_15Minutes", "type": "LONG"},
    {"name": "higher_highs_lows_15Minutes_negative", "type": "SHORT"},
    {"name": "ema_crossover_pullback_5Minutes", "type": "LONG"},
    {"name": "ema_crossover_pullback_5Minutes_negative", "type": "SHORT"},
    {"name": "ema_crossover_pullback_15Minutes", "type": "LONG"},
    {"name": "ema_crossover_pullback_15Minutes_negative", "type": "SHORT"},
    {"name": "ascending_channel_breakout_5Minutes", "type": "LONG"},
    {"name": "ascending_channel_breakout_5Minutes_negative", "type": "SHORT"},
    {"name": "ascending_channel_breakout_15Minutes", "type": "LONG"},
    {"name": "ascending_channel_breakout_15Minutes_negative", "type": "SHORT"},
    {"name": "trend_reversal_up_5Minutes", "type": "LONG"},
    {"name": "trend_reversal_up_5Minutes_negative", "type": "SHORT"},
    {"name": "trend_reversal_up_15Minutes", "type": "LONG"},
    {"name": "trend_reversal_up_15Minutes_negative", "type": "SHORT"},
    {"name": "double_bottom_breakout_5Minutes", "type": "LONG"},
    {"name": "double_bottom_breakout_5Minutes_negative", "type": "SHORT"},
    {"name": "double_bottom_breakout_15Minutes", "type": "LONG"},
    {"name": "double_bottom_breakout_15Minutes_negative", "type": "SHORT"},
    {"name": "falling_wedge_breakout_5Minutes", "type": "LONG"},
    {"name": "falling_wedge_breakout_5Minutes_negative", "type": "SHORT"},
    {"name": "falling_wedge_breakout_15Minutes", "type": "LONG"},
    {"name": "falling_wedge_breakout_15Minutes_negative", "type": "SHORT"},
    {"name": "volume_dryup_spike_5Minutes", "type": "LONG"},
    {"name": "volume_dryup_spike_5Minutes_negative", "type": "SHORT"},
    {"name": "volume_dryup_spike_15Minutes", "type": "LONG"},
    {"name": "volume_dryup_spike_15Minutes_negative", "type": "SHORT"},
    {"name": "ema_squeeze_breakout_5Minutes", "type": "LONG"},
    {"name": "ema_squeeze_breakout_5Minutes_negative", "type": "SHORT"},
    {"name": "ema_squeeze_breakout_15Minutes", "type": "LONG"},
    {"name": "ema_squeeze_breakout_15Minutes_negative", "type": "SHORT"},
    {"name": "oversold_bounce_reversal_5Minutes", "type": "LONG"},
    {"name": "oversold_bounce_reversal_5Minutes_negative", "type": "SHORT"},
    {"name": "oversold_bounce_reversal_15Minutes", "type": "LONG"},
    {"name": "oversold_bounce_reversal_15Minutes_negative", "type": "SHORT"},
    {"name": "consolidation_breakout_up_5Minutes", "type": "LONG"},
    {"name": "consolidation_breakout_up_5Minutes_negative", "type": "SHORT"},
    {"name": "consolidation_breakout_up_15Minutes", "type": "LONG"},
    {"name": "consolidation_breakout_up_15Minutes_negative", "type": "SHORT"},
    {"name": "bullish_abandoned_baby_5Minutes", "type": "LONG"},
    {"name": "bullish_abandoned_baby_5Minutes_negative", "type": "SHORT"},
    {"name": "bullish_abandoned_baby_15Minutes", "type": "LONG"},
    {"name": "bullish_abandoned_baby_15Minutes_negative", "type": "SHORT"},
    {"name": "bullish_breakaway_gap_5Minutes", "type": "LONG"},
    {"name": "bullish_breakaway_gap_5Minutes_negative", "type": "SHORT"},
    {"name": "bullish_breakaway_gap_15Minutes", "type": "LONG"},
    {"name": "bullish_breakaway_gap_15Minutes_negative", "type": "SHORT"},
    {"name": "bullish_railway_tracks_5Minutes", "type": "LONG"},
    {"name": "bullish_railway_tracks_5Minutes_negative", "type": "SHORT"},
    {"name": "bullish_railway_tracks_15Minutes", "type": "LONG"},
    {"name": "bullish_railway_tracks_15Minutes_negative", "type": "SHORT"},
    {"name": "inside_bar_breakout_up_5Minutes", "type": "LONG"},
    {"name": "inside_bar_breakout_up_5Minutes_negative", "type": "SHORT"},
    {"name": "inside_bar_breakout_up_15Minutes", "type": "LONG"},
    {"name": "inside_bar_breakout_up_15Minutes_negative", "type": "SHORT"},
    {"name": "bullish_pennant_breakout_5Minutes", "type": "LONG"},
    {"name": "bullish_pennant_breakout_5Minutes_negative", "type": "SHORT"},
    {"name": "bullish_pennant_breakout_15Minutes", "type": "LONG"},
    {"name": "bullish_pennant_breakout_15Minutes_negative", "type": "SHORT"},
    {"name": "ema21_pullback_bounce_5Minutes", "type": "LONG"},
    {"name": "ema21_pullback_bounce_5Minutes_negative", "type": "SHORT"},
    {"name": "ema21_pullback_bounce_15Minutes", "type": "LONG"},
    {"name": "ema21_pullback_bounce_15Minutes_negative", "type": "SHORT"},
    {"name": "ema50_reclaim_5Minutes", "type": "LONG"},
    {"name": "ema50_reclaim_5Minutes_negative", "type": "SHORT"},
    {"name": "ema50_reclaim_15Minutes", "type": "LONG"},
    {"name": "ema50_reclaim_15Minutes_negative", "type": "SHORT"},
    {"name": "higher_low_retest_breakout_5Minutes", "type": "LONG"},
    {"name": "higher_low_retest_breakout_5Minutes_negative", "type": "SHORT"},
    {"name": "higher_low_retest_breakout_15Minutes", "type": "LONG"},
    {"name": "higher_low_retest_breakout_15Minutes_negative", "type": "SHORT"},
    {"name": "volume_climax_reversal_up_5Minutes", "type": "LONG"},
    {"name": "volume_climax_reversal_up_5Minutes_negative", "type": "SHORT"},
    {"name": "volume_climax_reversal_up_15Minutes", "type": "LONG"},
    {"name": "volume_climax_reversal_up_15Minutes_negative", "type": "SHORT"},
    {"name": "triple_bottom_breakout_5Minutes", "type": "LONG"},
    {"name": "triple_bottom_breakout_5Minutes_negative", "type": "SHORT"},
    {"name": "triple_bottom_breakout_15Minutes", "type": "LONG"},
    {"name": "triple_bottom_breakout_15Minutes_negative", "type": "SHORT"},
    {"name": "range_shift_up_5Minutes", "type": "LONG"},
    {"name": "range_shift_up_5Minutes_negative", "type": "SHORT"},
    {"name": "range_shift_up_15Minutes", "type": "LONG"},
    {"name": "range_shift_up_15Minutes_negative", "type": "SHORT"},
    {"name": "ma_stack_trend_long_5Minutes", "type": "LONG"},
    {"name": "ma_stack_trend_long_5Minutes_negative", "type": "SHORT"},
    {"name": "ma_stack_trend_long_15Minutes", "type": "LONG"},
    {"name": "ma_stack_trend_long_15Minutes_negative", "type": "SHORT"},
    {"name": "vcp_breakout_long_5Minutes", "type": "LONG"},
    {"name": "vcp_breakout_long_5Minutes_negative", "type": "SHORT"},
    {"name": "vcp_breakout_long_15Minutes", "type": "LONG"},
    {"name": "vcp_breakout_long_15Minutes_negative", "type": "SHORT"},
    {"name": "bearish_engulfing_5Minutes", "type": "SHORT"},
    {"name": "bearish_engulfing_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_engulfing_15Minutes", "type": "SHORT"},
    {"name": "bearish_engulfing_15Minutes_negative", "type": "LONG"},
    {"name": "shooting_star_5Minutes", "type": "SHORT"},
    {"name": "shooting_star_5Minutes_negative", "type": "LONG"},
    {"name": "shooting_star_15Minutes", "type": "SHORT"},
    {"name": "shooting_star_15Minutes_negative", "type": "LONG"},
    {"name": "evening_star_5Minutes", "type": "SHORT"},
    {"name": "evening_star_5Minutes_negative", "type": "LONG"},
    {"name": "evening_star_15Minutes", "type": "SHORT"},
    {"name": "evening_star_15Minutes_negative", "type": "LONG"},
    {"name": "three_black_crows_5Minutes", "type": "SHORT"},
    {"name": "three_black_crows_5Minutes_negative", "type": "LONG"},
    {"name": "three_black_crows_15Minutes", "type": "SHORT"},
    {"name": "three_black_crows_15Minutes_negative", "type": "LONG"},
    {"name": "dark_cloud_cover_5Minutes", "type": "SHORT"},
    {"name": "dark_cloud_cover_5Minutes_negative", "type": "LONG"},
    {"name": "dark_cloud_cover_15Minutes", "type": "SHORT"},
    {"name": "dark_cloud_cover_15Minutes_negative", "type": "LONG"},
    {"name": "tweezer_top_5Minutes", "type": "SHORT"},
    {"name": "tweezer_top_5Minutes_negative", "type": "LONG"},
    {"name": "tweezer_top_15Minutes", "type": "SHORT"},
    {"name": "tweezer_top_15Minutes_negative", "type": "LONG"},
    {"name": "hanging_man_5Minutes", "type": "SHORT"},
    {"name": "hanging_man_5Minutes_negative", "type": "LONG"},
    {"name": "hanging_man_15Minutes", "type": "SHORT"},
    {"name": "hanging_man_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_harami_5Minutes", "type": "SHORT"},
    {"name": "bearish_harami_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_harami_15Minutes", "type": "SHORT"},
    {"name": "bearish_harami_15Minutes_negative", "type": "LONG"},
    {"name": "falling_three_methods_5Minutes", "type": "SHORT"},
    {"name": "falling_three_methods_5Minutes_negative", "type": "LONG"},
    {"name": "falling_three_methods_15Minutes", "type": "SHORT"},
    {"name": "falling_three_methods_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_marubozu_5Minutes", "type": "SHORT"},
    {"name": "bearish_marubozu_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_marubozu_15Minutes", "type": "SHORT"},
    {"name": "bearish_marubozu_15Minutes_negative", "type": "LONG"},
    {"name": "three_inside_down_5Minutes", "type": "SHORT"},
    {"name": "three_inside_down_5Minutes_negative", "type": "LONG"},
    {"name": "three_inside_down_15Minutes", "type": "SHORT"},
    {"name": "three_inside_down_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_kicker_5Minutes", "type": "SHORT"},
    {"name": "bearish_kicker_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_kicker_15Minutes", "type": "SHORT"},
    {"name": "bearish_kicker_15Minutes_negative", "type": "LONG"},
    {"name": "descending_channel_breakdown_5Minutes", "type": "SHORT"},
    {"name": "descending_channel_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "descending_channel_breakdown_15Minutes", "type": "SHORT"},
    {"name": "descending_channel_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "lower_highs_lows_5Minutes", "type": "SHORT"},
    {"name": "lower_highs_lows_5Minutes_negative", "type": "LONG"},
    {"name": "lower_highs_lows_15Minutes", "type": "SHORT"},
    {"name": "lower_highs_lows_15Minutes_negative", "type": "LONG"},
    {"name": "ema_channel_down_5Minutes", "type": "SHORT"},
    {"name": "ema_channel_down_5Minutes_negative", "type": "LONG"},
    {"name": "ema_channel_down_15Minutes", "type": "SHORT"},
    {"name": "ema_channel_down_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_ema_cross_5Minutes", "type": "SHORT"},
    {"name": "bearish_ema_cross_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_ema_cross_15Minutes", "type": "SHORT"},
    {"name": "bearish_ema_cross_15Minutes_negative", "type": "LONG"},
    {"name": "double_top_5Minutes", "type": "SHORT"},
    {"name": "double_top_5Minutes_negative", "type": "LONG"},
    {"name": "double_top_15Minutes", "type": "SHORT"},
    {"name": "double_top_15Minutes_negative", "type": "LONG"},
    {"name": "ema_crossover_down_5Minutes", "type": "SHORT"},
    {"name": "ema_crossover_down_5Minutes_negative", "type": "LONG"},
    {"name": "ema_crossover_down_15Minutes", "type": "SHORT"},
    {"name": "ema_crossover_down_15Minutes_negative", "type": "LONG"},
    {"name": "lower_high_breakdown_5Minutes", "type": "SHORT"},
    {"name": "lower_high_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "lower_high_breakdown_15Minutes", "type": "SHORT"},
    {"name": "lower_high_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "trend_channel_down_5Minutes", "type": "SHORT"},
    {"name": "trend_channel_down_5Minutes_negative", "type": "LONG"},
    {"name": "trend_channel_down_15Minutes", "type": "SHORT"},
    {"name": "trend_channel_down_15Minutes_negative", "type": "LONG"},
    {"name": "bear_flag_5Minutes", "type": "SHORT"},
    {"name": "bear_flag_5Minutes_negative", "type": "LONG"},
    {"name": "bear_flag_15Minutes", "type": "SHORT"},
    {"name": "bear_flag_15Minutes_negative", "type": "LONG"},
    {"name": "shooting_star_breakdown_5Minutes", "type": "SHORT"},
    {"name": "shooting_star_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "shooting_star_breakdown_15Minutes", "type": "SHORT"},
    {"name": "shooting_star_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "double_top_breakdown_5Minutes", "type": "SHORT"},
    {"name": "double_top_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "double_top_breakdown_15Minutes", "type": "SHORT"},
    {"name": "double_top_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_ema_cross_drop_5Minutes", "type": "SHORT"},
    {"name": "bearish_ema_cross_drop_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_ema_cross_drop_15Minutes", "type": "SHORT"},
    {"name": "bearish_ema_cross_drop_15Minutes_negative", "type": "LONG"},
    {"name": "falling_wedge_breakdown_5Minutes", "type": "SHORT"},
    {"name": "falling_wedge_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "falling_wedge_breakdown_15Minutes", "type": "SHORT"},
    {"name": "falling_wedge_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "trend_exhaustion_breakdown_5Minutes", "type": "SHORT"},
    {"name": "trend_exhaustion_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "trend_exhaustion_breakdown_15Minutes", "type": "SHORT"},
    {"name": "trend_exhaustion_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "volatility_squeeze_breakdown_5Minutes", "type": "SHORT"},
    {"name": "volatility_squeeze_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "volatility_squeeze_breakdown_15Minutes", "type": "SHORT"},
    {"name": "volatility_squeeze_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "rsi_divergence_breakdown_5Minutes", "type": "SHORT"},
    {"name": "rsi_divergence_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "rsi_divergence_breakdown_15Minutes", "type": "SHORT"},
    {"name": "rsi_divergence_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "overbought_pullback_reversal_5Minutes", "type": "SHORT"},
    {"name": "overbought_pullback_reversal_5Minutes_negative", "type": "LONG"},
    {"name": "overbought_pullback_reversal_15Minutes", "type": "SHORT"},
    {"name": "overbought_pullback_reversal_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_abandoned_baby_5Minutes", "type": "SHORT"},
    {"name": "bearish_abandoned_baby_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_abandoned_baby_15Minutes", "type": "SHORT"},
    {"name": "bearish_abandoned_baby_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_breakaway_gap_5Minutes", "type": "SHORT"},
    {"name": "bearish_breakaway_gap_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_breakaway_gap_15Minutes", "type": "SHORT"},
    {"name": "bearish_breakaway_gap_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_railway_tracks_5Minutes", "type": "SHORT"},
    {"name": "bearish_railway_tracks_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_railway_tracks_15Minutes", "type": "SHORT"},
    {"name": "bearish_railway_tracks_15Minutes_negative", "type": "LONG"},
    {"name": "inside_bar_breakdown_5Minutes", "type": "SHORT"},
    {"name": "inside_bar_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "inside_bar_breakdown_15Minutes", "type": "SHORT"},
    {"name": "inside_bar_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "bearish_pennant_breakdown_5Minutes", "type": "SHORT"},
    {"name": "bearish_pennant_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "bearish_pennant_breakdown_15Minutes", "type": "SHORT"},
    {"name": "bearish_pennant_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "ema21_pullback_reject_5Minutes", "type": "SHORT"},
    {"name": "ema21_pullback_reject_5Minutes_negative", "type": "LONG"},
    {"name": "ema21_pullback_reject_15Minutes", "type": "SHORT"},
    {"name": "ema21_pullback_reject_15Minutes_negative", "type": "LONG"},
    {"name": "ema50_failure_5Minutes", "type": "SHORT"},
    {"name": "ema50_failure_5Minutes_negative", "type": "LONG"},
    {"name": "ema50_failure_15Minutes", "type": "SHORT"},
    {"name": "ema50_failure_15Minutes_negative", "type": "LONG"},
    {"name": "lower_high_retest_breakdown_5Minutes", "type": "SHORT"},
    {"name": "lower_high_retest_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "lower_high_retest_breakdown_15Minutes", "type": "SHORT"},
    {"name": "lower_high_retest_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "volume_climax_reversal_down_5Minutes", "type": "SHORT"},
    {"name": "volume_climax_reversal_down_5Minutes_negative", "type": "LONG"},
    {"name": "volume_climax_reversal_down_15Minutes", "type": "SHORT"},
    {"name": "volume_climax_reversal_down_15Minutes_negative", "type": "LONG"},
    {"name": "triple_top_breakdown_5Minutes", "type": "SHORT"},
    {"name": "triple_top_breakdown_5Minutes_negative", "type": "LONG"},
    {"name": "triple_top_breakdown_15Minutes", "type": "SHORT"},
    {"name": "triple_top_breakdown_15Minutes_negative", "type": "LONG"},
    {"name": "range_shift_down_5Minutes", "type": "SHORT"},
    {"name": "range_shift_down_5Minutes_negative", "type": "LONG"},
    {"name": "range_shift_down_15Minutes", "type": "SHORT"},
    {"name": "range_shift_down_15Minutes_negative", "type": "LONG"},
    {"name": "ma_stack_trend_short_5Minutes", "type": "SHORT"},
    {"name": "ma_stack_trend_short_5Minutes_negative", "type": "LONG"},
    {"name": "ma_stack_trend_short_15Minutes", "type": "SHORT"},
    {"name": "ma_stack_trend_short_15Minutes_negative", "type": "LONG"},
    {"name": "vcp_breakdown_short_5Minutes", "type": "SHORT"},
    {"name": "vcp_breakdown_short_5Minutes_negative", "type": "LONG"},
    {"name": "vcp_breakdown_short_15Minutes", "type": "SHORT"},
    {"name": "vcp_breakdown_short_15Minutes_negative", "type": "LONG"},
]
#****************************************************************************************************************
#****************************************************************************************************************
# Step 2: Load latest candles from CSV

def load_candles_from_csv(path=CSV_PATH, limit=390):
    """
    טוען את קובץ ה־CSV של הנרות ושומר רק את X האחרונים
    """
    if not os.path.exists(path):
        print(f"❌ CSV file not found: {path}")
        return None

    try:
        df = pd.read_csv(path)

        # שמירה רק על העמודות הדרושות
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns in CSV: {', '.join(missing_cols)}")
            return None

        df = df[required_cols].copy()

        # המרות בטוחות כדי להתגבר על שורות פגומות
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # זריקה של שורות פגומות במקום לעצור את הבוט
        before = len(df)
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"⚠️ Dropped {dropped} corrupt candle row(s) from CSV.")

        # מיון לפי זמן, שמירה על X האחרונים
        df = df.sort_values('timestamp').tail(limit).reset_index(drop=True)
        return df

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None
def load_candles_from_api(limit=390):
    """
    Placeholder for a future real broker/API integration.

    Never fabricate candles: random OHLCV data can create false signals and
    damage both training results and live bot behavior.
    """
    log_alert(
        "WARN",
        "No broker/API candle integration is configured. Run generate_shared_candles_stocks.py to create shared_candles.csv.",
        key="api_not_configured",
    )
    return None


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


def load_unified_candles(limit=390, interval_seconds: int = 60, *, return_source: bool = False):
    """
    Prefer the shared TTP JSONL feed if present, otherwise fallback to CSV/API.
    """
    if os.path.exists(TTP_FEED_JSONL):
        source = "TTP JSON feed"
        try:
            df = load_ttp_jsonl_to_df(TTP_FEED_JSONL, interval_seconds=interval_seconds, limit=limit, symbol=SYMBOL)
        except Exception as exc:
            log_alert("WARN", f"Failed loading TTP feed: {exc}", key="load_ttp_feed")
            df = None
        df = validate_candles_df(df, source=source, min_rows=2)
        if df is not None and len(df) > 0:
            return (df, source) if return_source else df
    if os.path.exists(CSV_PATH):
        source = "CSV"
        df = load_candles_from_csv(CSV_PATH, limit=limit)
        df = validate_candles_df(df, source=source, min_rows=2)
        if df is not None and len(df) > 0:
            return (df, source) if return_source else df
    source = "API"
    df = load_candles_from_api(limit=limit)
    df = validate_candles_df(df, source=source, min_rows=2)
    return (df, source) if return_source else df
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
        log_alert(
            "ERROR",
            f"Failed reading {label} file: {path} ({exc}).",
            key=f"json_read:{path}",
        )
        return default_value

    if expect_type is not None and not isinstance(data, expect_type):
        label = purpose or "json"
        log_alert(
            "WARN",
            f"{label} file has unexpected structure: {path}. Resetting to defaults.",
            key=f"json_type:{path}",
        )
        safe_write_json(path, default_value, purpose=label)
        return default_value
    return data


def safe_write_json(path: str, payload, *, purpose: str | None = None) -> bool:
    """Write JSON atomically; returns True on success."""
    try:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        label = purpose or "json"
        log_alert(
            "ERROR",
            f"Failed writing {label} file: {path} ({exc}).",
            key=f"json_write:{path}",
        )
        return False


def init_file(path, default_value):
    """
    יוצר קובץ JSON אם הוא לא קיים
    """
    if not os.path.exists(path):
        if safe_write_json(path, default_value, purpose="init"):
            print(f"📁 Created missing file: {path}")

def initialize_all_files():
    """
    מוודא שכל קובצי JSON קיימים. אם חסרים – ייווצרו עם ערך ברירת מחדל.
    """
    init_file(CAPITAL_PATH, DEFAULT_CAPITAL_PAYLOAD)
    init_file(REINFORCEMENT_PATH, DEFAULT_REINFORCEMENT_PAYLOAD)
    init_file(PENDING_PATTERN_PATH, DEFAULT_PENDING_PATTERN)  # None = אין תבנית ממתינה
    # Bot71 keeps a closed-trade audit log for review and debugging.
    init_file(CLOSED_LOG_PATH, DEFAULT_CLOSED_LOG)

# === CAPITAL ===
def load_capital():
    if not os.path.exists(CAPITAL_PATH):
        log_alert("WARN", "CAPITAL file missing. Reinitializing...", key="capital_missing")
        safe_write_json(CAPITAL_PATH, DEFAULT_CAPITAL_PAYLOAD, purpose="capital")
    data = safe_read_json(CAPITAL_PATH, DEFAULT_CAPITAL_PAYLOAD, expect_type=dict, purpose="capital")
    try:
        capital = float(data.get("capital", DEFAULT_CAPITAL))
        if not np.isfinite(capital):
            raise ValueError("capital is not finite")
        return capital
    except Exception:
        log_alert("WARN", "Capital value invalid; using default.", key="capital_invalid")
        safe_write_json(CAPITAL_PATH, DEFAULT_CAPITAL_PAYLOAD, purpose="capital")
        return float(DEFAULT_CAPITAL)

def save_capital(value):
    try:
        capital = float(value)
        if not np.isfinite(capital):
            raise ValueError("capital is not finite")
    except Exception:
        log_alert("WARN", "Attempted to save invalid capital value. Skipping write.", key="capital_save_invalid")
        return
    safe_write_json(CAPITAL_PATH, {'capital': capital}, purpose="capital")


def _strip_internal_fields(trade_data):
    """Remove internal-only keys (prefixed with '_') before persisting logs."""
    if isinstance(trade_data, dict):
        return {k: v for k, v in trade_data.items() if not str(k).startswith("_")}
    return trade_data


# === CLOSED TRADE LOG ===
def update_closed_trade_log(trade_data, path=CLOSED_LOG_PATH):
    """
    שומר לוג של עסקאות שנסגרו.
    """
    logs = safe_read_json(path, DEFAULT_CLOSED_LOG, expect_type=list, purpose="closed trades")
    if not isinstance(logs, list):
        logs = []
    logs.append(_strip_internal_fields(trade_data))
    if not safe_write_json(path, logs, purpose="closed trades"):
        log_alert("ERROR", "Error writing closed trade log.", key="closed_log_write")

# === PENDING PATTERN ===
def load_pending_pattern():
    return safe_read_json(
        PENDING_PATTERN_PATH,
        DEFAULT_PENDING_PATTERN,
        expect_type=(dict, type(None)),
        purpose="pending pattern",
    )

def save_pending_pattern(pattern_data):
    if not safe_write_json(PENDING_PATTERN_PATH, pattern_data, purpose="pending pattern"):
        log_alert("WARN", "Failed to persist pending pattern.", key="pending_pattern_write")

# Runtime files are initialized only when the bot is executed directly.
# initialize_all_files()
#****************************************************************************************************************
#****************************************************************************************************************
# Step 4: Trade management (enter / ladder / exit)

# Fees (fractions) for NET PnL calculations (Alpaca equities → $0 commissions)
FEE_BUY  = 0.0
FEE_SELL = 0.0


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
    מחזיר מילון {indicator_name: bool} מסונכרן לתבנית האינדיקטורים של התבנית.
    קלט אפשרי:
      1) {"overall": bool, "checks": {name: bool, ...}}
      2) dict שטוח של Booleans (ללא overall)
      3) ערך בוליאני פשוט → לא מסיקים לכל אינדיקטור; נחזיר {}.

    כללי:
    - רק אינדיקטורים שחזרו מפורשות ב־checks ייחשבו.
    - אינדיקטורים חסרים ימופו ל־False כשקיימת רשימת expected.
    - לעולם לא ממלאים אינדיקטורים מתוך "overall".
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

    # simple bool → לא מסיקים ערכים פר-אינדיקטור
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
    Meaning (for LONG): when profit >= 0.15% → TP=+0.35%, SL=+0.12% (in profit)
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
            #  - LONG  → profit is ABOVE entry (add fraction)
            #  - SHORT → profit is BELOW entry (subtract fraction)
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
        print(f"📈 Ladder → level {new_level} | TP {tp_price:.4f} | SL {sl_price:.4f} | SL profit now {format_percent(sl_profit)}")
    return open_trade


def enter_trade(df, breakout_idx: int, pattern_type: str, capital_data: dict, live=False, api=None, symbol=SYMBOL, context_key: str | None = None) -> dict | None:
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

    # --- Available capital (fallback to stored capital if not provided) ---
    try:
        available_capital = float(capital_data.get("capital", load_capital()))
    except Exception:
        available_capital = 0.0

    # Position sizing priority:
    # 1) explicit position_size (units)
    # 2) risk-based using risk_pct and SL distance
    # 3) allocation fraction fallback
    if "position_size" in capital_data:
        quantity = floor_shares(capital_data.get("position_size", 1.0))
    else:
        risk_pct = float(capital_data.get("risk_pct", 0.0))
        if risk_pct > 0:
            cap = available_capital
            try:
                sl_dist = abs((sl_price / entry_price) - 1.0)
                sl_dist = max(sl_dist, 1e-4)
                dollar_risk = cap * max(0.0, min(1.0, risk_pct))
                qty_est = dollar_risk / (entry_price * sl_dist)
                quantity = floor_shares(qty_est)
            except Exception:
                alloc_frac = float(capital_data.get("allocation_frac", 1.0))
                alloc_cap = available_capital * max(0.0, min(1.0, alloc_frac))
                # floor to whole units, accounting for entry-side fee
                entry_fee = float(FEE_BUY if side == "LONG" else FEE_SELL)
                per_unit = entry_price * (1.0 + entry_fee)
                max_afford = floor_shares(alloc_cap / per_unit) if per_unit > 0 else 0
                quantity = max_afford
        else:
            alloc_frac = float(capital_data.get("allocation_frac", 1.0))
            alloc_cap = available_capital * max(0.0, min(1.0, alloc_frac))
            # floor to whole units, accounting for entry-side fee
            entry_fee = float(FEE_BUY if side == "LONG" else FEE_SELL)
            per_unit = entry_price * (1.0 + entry_fee)
            max_afford = floor_shares(alloc_cap / per_unit) if per_unit > 0 else 0
            quantity = max_afford

    # Final affordability cap: ensure we never exceed available capital (incl. entry fee)
    entry_fee = float(FEE_BUY if side == "LONG" else FEE_SELL)
    per_unit_cost = entry_price * (1.0 + entry_fee)
    max_qty_allowed = floor_shares(available_capital / per_unit_cost) if per_unit_cost > 0 else 0
    try:
        quantity = min(floor_shares(quantity), max_qty_allowed)
    except Exception:
        quantity = max_qty_allowed

    if max_qty_allowed < 1 or quantity < 1:
        record_blocked_opportunity(side, entry_price, timestamp, "insufficient_capital", pattern=pattern_type, key=context_key)
        print(f"⛔ Not enough capital to open {side}: capital {available_capital:.2f}$, need ≥ {per_unit_cost:.2f}$ per unit (incl. fees)")
        return None

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

    # Save pending pattern (for visibility in UI/logs)
    save_pending_pattern({
        "pattern": pattern_type,
        "side": side,
        "key": context_key,
        "score_at_entry": None,
        "timestamp": str(timestamp),
        "symbol": symbol,
        "timeframe": f"{INTERVAL_MINUTES}m",
        "entry_price": entry_price
    })

    position_notional = entry_price * quantity
    print(
        f"🟢 Enter {side}\n"
        f"🔹 Entry Time : {timestamp}\n"
        f"🔹 Entry Price: {entry_price:.2f}\n"
        f"💵 Position    : {quantity:g} units  | Notional ≈ {position_notional:.2f}$\n"
        f"🎯 TP/SL      : TP {format_percent((tp_price/entry_price)-1)} | SL {format_percent((sl_price/entry_price)-1)} (ladder off until first threshold)\n"
        f"📐 Pattern    : {pattern_type}"
    )
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
        print(f"⚠️ Failed to compute NET PnL: {e}")
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
            print(f"⚠️ RL update failed: {e}")

    # Save closed trade log
    update_closed_trade_log({**open_trade})

    # Clear pending
    save_pending_pattern(None)

    # --- Persist capital update (always applies; includes fees via NET PnL) ---
    prev_capital = None
    new_capital = None
    try:
        prev_capital = float(load_capital())
        new_capital = round(prev_capital + float(pnl_dollars), 2)
        save_capital(new_capital)
    except Exception as e:
        print(f"⚠️ Failed to update capital_1.json: {e}")

    mode = "🔴 LIVE Trade" if live else "🟢 Simulated Trade"
    print(f"\n💼 {mode} closed: {'⬆️ LONG' if signal == 'LONG' else '⬇️ SHORT'}")
    print(f"🔹 Entry Price  : {entry_price:.2f}")
    print(f"🔹 Exit Price   : {exit_price:.2f}")
    print(f"💰 Net Result   : {pnl_dollars:.2f}$ ({format_percent(net_pnl_fraction)})")
    print(f"✅ Result Type  : {success}")
    if prev_capital is not None and new_capital is not None:
        print(f"🏦 Capital Update : {prev_capital:.2f}$ → {new_capital:.2f}$")
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
            print(f"⚠️ Failed to compute rsi_14: {e}")

    # Common EMAs used by channel/crossover rules
    # NOTE: include 150 because some detectors/confirmers (e.g., golden_cross, some LONG confirmers) rely on ema_150
    ema_periods = [9, 21, 50, 150, 200]
    for p in ema_periods:
        col = f'ema_{p}'
        if col not in df.columns:
            try:
                df[col] = _compute_ema(df['close'], period=p)
            except Exception as e:
                print(f"⚠️ Failed to compute {col}: {e}")

    # Volume SMA(20) for volume spike checks
    if 'volume_sma_20' not in df.columns:
        try:
            df['volume_sma_20'] = pd.to_numeric(df['volume'], errors='coerce').rolling(window=20, min_periods=1).mean()
        except Exception as e:
            print(f"⚠️ Failed to compute volume_sma_20: {e}")

    # Close-to-close volatility proxy: rolling std of returns (20)
    if 'volatility_20' not in df.columns:
        try:
            ret = pd.to_numeric(df['close'], errors='coerce').pct_change()
            df['volatility_20'] = ret.rolling(window=20, min_periods=5).std()
        except Exception as e:
            print(f"⚠️ Failed to compute volatility_20: {e}")

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
            print(f"⚠️ Failed to compute atr_14: {e}")

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
# Heartbeat printer – one line status every loop
# ============================================================
def print_heartbeat(
    current_price: float,
    open_trade: dict | None,
    *,
    session: dict | None = None,
    data_delay_min: float | None = None,
):
    """Print a short status line every minute so the loop feels alive."""
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
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
            if mtc is not None:
                session_note = f"NY {ny_txt} ({mtc:.0f}m to close) | IL {il_txt}"
            else:
                session_note = f"NY {ny_txt} | IL {il_txt}"
        else:
            session_note = f"Market closed | NY {ny_txt} | IL {il_txt}"

    if open_trade and open_trade.get("status") == "open":
        side = open_trade.get("signal", "LONG")
        entry = float(open_trade.get("entry_price", current_price))
        level = int(open_trade.get("current_level", 0))
        tp = open_trade.get("tp_price")
        sl = open_trade.get("sl_price")
        move = compute_profit_move_fraction(entry, current_price, side)
        qty = float(open_trade.get("quantity", 1.0))
        gross_dollars = move * entry * qty
        arrow = "⬇️ SHORT" if side == "SHORT" else "⬆️ LONG"
        tp_s = f"{tp:.2f}" if isinstance(tp, (int, float)) else "n/a"
        sl_s = f"{sl:.2f}" if isinstance(sl, (int, float)) else "n/a"
        parts = [
            f"⏱ {now_str}",
            f"Price {current_price:.2f}",
            f"{arrow} @{entry:.2f}",
            f"Lv {level}",
            f"Δ {format_percent(move)}",
            f"TP {tp_s}",
            f"SL {sl_s}",
            f"PnL(gross) {gross_dollars:+.2f}$",
        ]
        if data_delay_min is not None:
            parts.append(f"delay {data_delay_min:.1f}m")
        if session_note:
            parts.append(session_note)
        print(" | ".join(parts))
    else:
        parts = [
            f"⏱ {now_str}",
            f"Price {current_price:.2f}",
            "No open trade",
        ]
        if data_delay_min is not None:
            parts.append(f"delay {data_delay_min:.1f}m")
        if session_note:
            parts.append(session_note)
        print(" | ".join(parts))

# ============================================================
# Glue – detect → confirm → key → ensure → enter trade
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
        print(f"❌ Error in {cname}: {e}")
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
def process_signals_and_maybe_enter(
    df,
    capital_data,
    live=False,
    api=None,
    symbol=SYMBOL,
    minutes_to_close: float | None = None,
):
    if df is None or len(df) < 2:
        print("⚠️ DataFrame is too small to detect signals.")
        return None

    if minutes_to_close is None:
        try:
            minutes_to_close = market_session_status().get("minutes_to_close")
        except Exception:
            minutes_to_close = None

    # Ensure minimal indicators to prevent KeyErrors inside pattern detectors
    df = ensure_min_indicators(df)

    try:
        detectors = detect_Graphical_patterns(df)
    except Exception as exc:
        log_exception("Pattern detection failed; skipping this tick.", exc, key="detect_patterns")
        return None
    if not isinstance(detectors, dict):
        log_alert("ERROR", "Pattern detectors returned invalid output; skipping.", key="detect_patterns_type")
        return None
    last_idx = len(df) - 1
    last_close = float(df['close'].iloc[-1])
    last_ts = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(timezone.utc).isoformat()

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

    candidates: list[dict] = []

    for item in pattern_list:
        name = item.get("name")
        ptype = item.get("type")
        series = detectors.get(name)
        if series is None:
            continue

        # Check only the last candle for a signal
        try:
            last_val = bool(series.iloc[-1])
        except Exception:
            last_val = bool(series[-1]) if isinstance(series, (list, np.ndarray)) else False

        if last_val:
            side = get_pattern_side(name)

            # --- Regime filters for safer entry ---
            # Trend filter vs EMA200 + slope
            if ENABLE_TREND_FILTERS:
                try:
                    if ema200_val is not None:
                        slope_sym = '↗' if slope_sign > 0 else ('↘' if slope_sign < 0 else '→')
                        band = float(TREND_BAND_PCT)
                        # Softer blocking: block only if BOTH price is clearly on the opposite side beyond band AND slope opposes
                        if side == 'LONG':
                            long_block = (last_close <= ema200_val * (1 - band)) and (slope_sign < 0)
                            if long_block:
                                record_blocked_opportunity(side, last_close, last_ts, "trend_filter_against", pattern=name)
                                print(f"⛔ Skip ({side}) '{name}': Against trend | Price {last_close:.2f} ≤ EMA200 {ema200_val:.2f} (>{band*100:.2f}% away) and slope {slope_sym}")
                                continue
                        else:  # SHORT
                            short_block = (last_close >= ema200_val * (1 + band)) and (slope_sign > 0)
                            if short_block:
                                record_blocked_opportunity(side, last_close, last_ts, "trend_filter_against", pattern=name)
                                print(f"⛔ Skip ({side}) '{name}': Against trend | Price {last_close:.2f} ≥ EMA200 {ema200_val:.2f} (>{band*100:.2f}% away) and slope {slope_sym}")
                                continue
                except Exception:
                    pass

            # Volatility filter (ATR/close within band)
            if ENABLE_VOL_FILTER and vol_ratio is not None:
                if not (VOL_MIN <= vol_ratio <= VOL_MAX):
                    try:
                        be = ((1.0 + float(FEE_BUY)) / (1.0 - float(FEE_SELL))) - 1.0  # gross break-even vs fees
                    except Exception:
                        be = 0.002
                    vol_pct = vol_ratio * 100.0
                    min_pct = VOL_MIN * 100.0
                    max_pct = VOL_MAX * 100.0
                    be_pct  = be * 100.0
                    if vol_ratio < VOL_MIN:
                        record_blocked_opportunity(side, last_close, last_ts, "volatility_too_low", pattern=name)
                        print(f"⛔ Skip ({side}) '{name}': Market too quiet | Volatility ~{vol_pct:.2f}%/bar < {min_pct:.2f}% (needs ≈{be_pct:.2f}% just to cover fees)")
                    else:
                        record_blocked_opportunity(side, last_close, last_ts, "volatility_too_high", pattern=name)
                        print(f"⛔ Skip ({side}) '{name}': Excessive volatility | ~{vol_pct:.2f}%/bar > {max_pct:.2f}% (high whipsaw risk)")
                    continue

            # EMA50/200 spread filter
            if ENABLE_SPREAD_FILTER and spread_ratio is not None:
                if not (SPREAD_MIN <= spread_ratio <= SPREAD_MAX):
                    spread_pct = spread_ratio * 100.0
                    min_sp = SPREAD_MIN * 100.0
                    max_sp = SPREAD_MAX * 100.0
                    if spread_ratio < SPREAD_MIN:
                        record_blocked_opportunity(side, last_close, last_ts, "spread_too_narrow", pattern=name)
                        print(f"⛔ Skip ({side}) '{name}': Choppy market | EMA50/200 spread {spread_pct:.2f}% < {min_sp:.2f}% (likely noise, no follow-through)")
                    else:
                        record_blocked_opportunity(side, last_close, last_ts, "spread_too_wide", pattern=name)
                        print(f"⛔ Skip ({side}) '{name}': Extended trend | EMA50/200 spread {spread_pct:.2f}% > {max_sp:.2f}% (late entry risk)")
                    continue
            # Confirmation is informative only (do not gate on it)
            overall, checks = normalize_confirmation_result(name, df, last_idx)
            outcomes = build_indicator_outcomes(name, checks if checks else (overall if overall is not None else False))
            valid_checks = sum(1 for v in outcomes.values() if bool(v))
            total_checks = len(outcomes)

            # Log detection and indicator status with breakdown (centralized renderer)
            try:
                render_snapshot(df, open_trade=None, checks=outcomes if outcomes else None, pattern_name=name)
            except Exception:
                if outcomes:
                    trues = sum(1 for v in outcomes.values() if bool(v))
                    falses = total_checks - trues
                    print(f"🔬 Breakdown: {trues}✔ / {falses}✖ → " + ", ".join(f"{k}={'✔' if v else '✖'}" for k, v in outcomes.items()))
                else:
                    print("🔬 Breakdown: n/a (no checks returned)")

            # Reinforcement gating by pattern key (must exist with a score)
            try:
                key = build_pattern_key(name, outcomes)
            except Exception as exc:
                log_alert("WARN", f"Failed to build reinforcement key for {name}: {exc}", key="reinforcement_key")
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_key_missing", pattern=name)
                print(f"⛔ Skip ({side}) '{name}': no reinforcement score key.")
                continue

            scores = rm_load_scores()
            rec = scores.get(key) if isinstance(scores, dict) else None
            if not isinstance(rec, dict):
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_score_missing", pattern=name, key=key)
                print(f"⛔ Skip ({side}) '{name}': no reinforcement score for key '{key}'.")
                continue
            score_val = rec.get("score", None)
            if score_val is None:
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_score_missing", pattern=name, key=key)
                print(f"⛔ Skip ({side}) '{name}': missing reinforcement score for key '{key}'.")
                continue
            try:
                score = float(score_val)
            except Exception:
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_score_invalid", pattern=name, key=key)
                print(f"⛔ Skip ({side}) '{name}': invalid reinforcement score for key '{key}'.")
                continue
            try:
                wins = int(rec.get("wins", 0))
            except Exception:
                wins = 0
            try:
                losses = int(rec.get("losses", 0))
            except Exception:
                losses = 0
            total_trades = max(0, wins + losses)
            win_rate = (wins / total_trades) if total_trades > 0 else 0.0
            avg_duration_val = None
            if minutes_to_close is not None and minutes_to_close > 0:
                try:
                    avg_duration_val = float(rec.get("avg_duration_min", None))
                except Exception:
                    avg_duration_val = None
            if avg_duration_val is not None and avg_duration_val > minutes_to_close:
                record_blocked_opportunity(side, last_close, last_ts, "avg_duration_gt_time_to_close", pattern=name, key=key)
                print(
                    f"⛔ Skip ({side}) '{name}': avg duration {avg_duration_val:.1f}m > time to close {minutes_to_close:.1f}m"
                )
                continue
            if score < 0:
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_score_negative", pattern=name, key=key)
                print(f"⛔ Skip ({side}) '{name}': reinforcement score {score:.2f} is negative (history {wins}W/{losses}L)")
                continue
            if score <= RL_BLOCK_SCORE:
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_score_blocked", pattern=name, key=key)
                print(f"⛔ Skip ({side}) '{name}': reinforcement score {score:.2f} ≤ {RL_BLOCK_SCORE:.2f} (history {wins}W/{losses}L)")
                continue

            if score <= MIN_ENTRY_SCORE:
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_score_below_entry_min", pattern=name, key=key)
                print(
                    f"⛔ Skip ({side}) '{name}': score {score:.2f} ≤ {MIN_ENTRY_SCORE:.0f} "
                    f"(needs > {MIN_ENTRY_SCORE:.0f}) | history {wins}W/{losses}L"
                )
                continue
            if total_trades < MIN_ENTRY_TRADES:
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_too_few_trades", pattern=name, key=key)
                print(
                    f"⛔ Skip ({side}) '{name}': only {total_trades} trade(s); need ≥{MIN_ENTRY_TRADES} "
                    f"to validate win rate"
                )
                continue
            if win_rate < MIN_ENTRY_WIN_RATE:
                record_blocked_opportunity(side, last_close, last_ts, "reinforcement_win_rate_low", pattern=name, key=key)
                print(
                    f"⛔ Skip ({side}) '{name}': win rate {win_rate*100:.1f}% < {MIN_ENTRY_WIN_RATE*100:.0f}% "
                    f"(history {wins}W/{losses}L)"
                )
                continue

            # Simple risk scaling based on confirmation strength
            risk_pct = RISK_PCT_HIGH if total_checks and valid_checks == total_checks else RISK_PCT_LOW

            # ATR-based SL override (optional)
            sl_override_frac = None
            try:
                if vol_ratio is not None:
                    sl_override_frac = max(0.004, min(0.02, 0.7 * vol_ratio))
            except Exception:
                sl_override_frac = None

            candidates.append({
                "name": name,
                "side": side,
                "key": key,
                "risk_pct": risk_pct,
                "sl_override_frac": sl_override_frac,
                "valid_checks": valid_checks,
                "total_checks": total_checks,
                "score": score,
            })

    if not candidates:
        print("⏱ No confirmed signal on the last candle.")
        return None

    score_key = lambda c: (c.get("score", float("-inf")), c.get("valid_checks", 0))
    pick = max(candidates, key=score_key)
    better_block = find_recent_better_block(pick.get("side"), last_close, last_ts)
    if better_block:
        blocked_price = float(better_block.get("price"))
        blocked_ts = better_block.get("timestamp")
        age_min = float(better_block.get("age_min", 0.0))
        reason = better_block.get("reason", "unknown")
        blocked_pattern = better_block.get("pattern")
        try:
            blocked_ts_txt = blocked_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            blocked_ts_txt = str(blocked_ts)
        pattern_txt = f" | blocked_pattern={blocked_pattern}" if blocked_pattern else ""
        print(
            f"⛔ Skip ({pick.get('side')}) '{pick['name']}': better price was blocked "
            f"{age_min:.1f}m ago at {blocked_price:.2f} (current {last_close:.2f}) | "
            f"{blocked_ts_txt} | reason={reason}{pattern_txt}"
        )
        return None

    print(f"✅ Entering with highest score pattern '{pick['name']}'.")

    trade = enter_trade(
        df=df,
        breakout_idx=last_idx,
        pattern_type=pick["name"],
        capital_data={
            **(capital_data or {}),
            "risk_pct": pick.get("risk_pct", RISK_PCT_LOW),
            "sl_override_frac": pick.get("sl_override_frac"),
        },
        live=live,
        api=api,
        symbol=symbol,
        context_key=pick.get("key"),
    )
    return trade

#****************************************************************************************************************
#****************************************************************************************************************
# Step 5: Continuous runner (loops, CSV first, API fallback)

if __name__ == "__main__":
    try:
        initialize_all_files()

        initial_df, data_source = load_unified_candles(
            limit=390,
            interval_seconds=int(INTERVAL_MINUTES * 60),
            return_source=True,
        )
        data_source = data_source or "Unknown"
        print("\n🚀 Bot started (continuous mode)")
        print(f"💾 Data source: {data_source}")
        print(f"💼 Capital: {load_capital():,.0f}$")
        print(f"⚙️ Fees: buy {FEE_BUY*100:.2f}% | sell {FEE_SELL*100:.2f}%")
        print(f"📚 Patterns loaded: {len(pattern_list)} (LONG:{sum(1 for p in pattern_list if p['type']=='LONG')} | SHORT:{sum(1 for p in pattern_list if p['type']=='SHORT')})\n")

        day_high_txt = "N/A"
        day_low_txt = "N/A"
        last_change_txt = "N/A"
        if initial_df is not None and len(initial_df) > 0:
            initial_df = filter_regular_market_hours(initial_df)
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
                print(f"⚠️ Failed to compute intraday stats: {e}")

        print(f"🧭 {SYMBOL}: day high {day_high_txt} | day low {day_low_txt} | last change {last_change_txt}\n")

        # Wiring consistency check: detectors ↔ confirmers ↔ rules
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
                print(f"⚠️ Missing confirmation functions for: {', '.join(missing_confirm)}")
            if missing_rules:
                print(f"⚠️ Missing indicator rules for: {', '.join(missing_rules)}")
        except Exception as e:
            print(f"⚠️ Wiring check failed: {e}")

        # Try restore an open trade from pending_pattern
        open_trade = None
        pending = load_pending_pattern()
        if isinstance(pending, dict) and pending.get('pattern'):
            entry_price = _safe_float(pending.get("entry_price"))
            if entry_price is None or entry_price <= 0:
                log_alert("WARN", "Pending pattern has invalid entry_price; skipping restore.", key="pending_restore")
            else:
                print("🗂 Restoring trade from pending_pattern.json…")
                qty_val = floor_shares(pending.get("quantity", 1)) or 1
                open_trade = {
                    "symbol": normalize_symbol(pending.get("symbol", SYMBOL)) or SYMBOL,
                    "signal": pending.get("side", "LONG"),
                    "pattern": pending.get("pattern"),
                    "entry_time": str(pending.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                    "entry_price": entry_price,
                    "quantity": qty_val,
                    "status": "open",
                    "current_level": 0,
                }

        # Main loop
        last_processed_ts = None  # prevent duplicate detection on the same candle
        last_seen_ts = None
        same_candle_streak = 0
        no_candle_warned = False
        loop_error_streak = 0
        while True:
            try:
                session = market_session_status()
                if not session.get("is_active_window"):
                    sleep_until_next_market_open(session)
                    last_processed_ts = None
                    last_seen_ts = None
                    same_candle_streak = 0
                    no_candle_warned = False
                    continue

                # Load candles
                limit = 390
                df, source = load_unified_candles(
                    limit=limit,
                    interval_seconds=int(INTERVAL_MINUTES * 60),
                    return_source=True,
                )

                if df is None or len(df) < 2:
                    log_alert(
                        "WARN",
                        f"Not enough candles from {source or 'data source'}. Retrying in 10s…",
                        key="not_enough_candles",
                    )
                    time.sleep(10)
                    continue

                # Keep only regular session bars (US market hours)
                try:
                    df = filter_regular_market_hours(df)
                except Exception as e:
                    log_alert("WARN", f"Failed to filter regular hours: {e}", key="filter_hours")

                if FILL_MISSING_CANDLES:
                    df = fill_small_candle_gaps(
                        df,
                        source=source or "data source",
                        interval_minutes=INTERVAL_MINUTES,
                    )
                    if df is not None:
                        df = df.tail(limit).reset_index(drop=True)

                if df is None or len(df) < 2:
                    log_alert(
                        "INFO",
                        "Waiting for regular US market session data (market likely closed). Sleeping 60s…",
                        key="market_closed_wait",
                        throttle_seconds=300,
                    )
                    time.sleep(60)
                    continue

                ts_series = None
                if 'timestamp' in df.columns:
                    try:
                        ts_series = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                        df['timestamp'] = ts_series.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                    except Exception:
                        ts_series = None

                if ts_series is None or ts_series.isna().all():
                    log_alert("WARN", "Timestamp parsing failed; retrying in 15s…", key="timestamp_parse")
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
                if data_delay_min is not None and data_delay_min > DATA_STALE_MINUTES:
                    log_alert(
                        "WARN",
                        f"Data is stale (~{data_delay_min:.1f}m). Waiting for fresher bars…",
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
                data_ok_for_entries = not data_gap_issue and not unfilled_gap_issue
                allow_new_entries = market_open_now and data_ok_for_entries and not (
                    minutes_to_close is not None and minutes_to_close <= ENTRY_BLOCK_BUFFER_MIN
                )

                if new_candle:
                    # Update/force-close open trade first (if exists)
                    if open_trade and open_trade.get("status") == "open":
                        if minutes_to_close is not None and minutes_to_close <= FORCED_EXIT_BUFFER_MIN:
                            print(f"⏰ Market closes in {minutes_to_close:.1f}m → forcing exit of open position.")
                            closed_trade = exit_trade(
                                open_trade,
                                exit_price=last_px,
                                timestamp=last_ts,
                                symbol=open_trade.get("symbol", SYMBOL),
                                live=False,
                            )
                            open_trade = None if closed_trade.get("status") == "closed" else closed_trade
                        if open_trade and open_trade.get("status") == "open":
                            open_trade, closed = on_tick_update_trade(
                                open_trade,
                                current_price=last_px,
                                timestamp=last_ts,
                                symbol=open_trade.get("symbol", SYMBOL),
                                live=False,
                            )
                            if closed:
                                open_trade = None

                    # If no open trade → try to enter by pattern on the last candle
                    if allow_new_entries and not (open_trade and open_trade.get("status") == "open"):
                        if last_processed_ts is None or last_ts != last_processed_ts:
                            capital_data = {"capital": load_capital(), "allocation_frac": 1.0}
                            trade = process_signals_and_maybe_enter(
                                df,
                                capital_data=capital_data,
                                live=False,
                                api=None,
                                symbol=SYMBOL,
                                minutes_to_close=minutes_to_close,
                            )
                            last_processed_ts = last_ts
                            if trade and trade.get("status") == "open":
                                open_trade = trade
                    else:
                        last_processed_ts = last_ts

                    # Heartbeat status line each loop
                    print_heartbeat(last_px, open_trade, session=session, data_delay_min=data_delay_min)
                    if not market_open_now:
                        print("⏸ US market is closed. Skipping entries until the next session window.")
                    elif not allow_new_entries:
                        if not data_ok_for_entries:
                            print("⏸ Entry gate closed due to data gap/unfilled candles.")
                        else:
                            print(f"⏸ Entry gate closed before market close (≥{ENTRY_BLOCK_BUFFER_MIN}m buffer).")
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
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
