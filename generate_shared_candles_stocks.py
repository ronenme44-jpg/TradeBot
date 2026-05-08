"""Collect intraday US equity candles into ``shared_candles.csv``.

This public collector uses yfinance because it requires no broker credentials.
The bot reads the generated CSV as its primary OHLCV input.
"""

import pandas as pd
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


# User-editable settings.
# SYMBOL must match the symbol used by tradeBot_main.py and tradeBot_Pattern_Test.py.
SYMBOL = "AAPL"

# The public version is tuned for 1-minute candles.
# If this is changed to "5m", the polling interval and yfinance interval update together.
INTERVAL_TYPE = "1m"
YF_INTERVAL = "1m" if INTERVAL_TYPE == "1m" else "5m"

# Both bots look for this exact CSV file in the repository directory.
OUTPUT_FILE = Path(__file__).resolve().parent / "shared_candles.csv"

# A regular US equity session has 390 one-minute candles.
MIN_CANDLES = 390

# Runtime controls. Polling faster than the candle interval is intentional:
# the loop wakes often, but it writes only when a new candle appears.
INTERVAL_SECONDS = 60 if INTERVAL_TYPE == "1m" else 300
POLL_INTERVAL_SECONDS = 5
NO_CANDLE_LOG_THROTTLE_SECONDS = 60
NO_CANDLE_WARN_MULTIPLIER = 2.0
MARKET_TZ = ZoneInfo("America/New_York")
LOCAL_TZ = ZoneInfo("Asia/Jerusalem")
MARKET_OPEN_MINUTE = 9 * 60 + 30
MARKET_CLOSE_MINUTE = 16 * 60
FREE_DATA_DELAY_MINUTES = 20
POST_CLOSE_GRACE_SECONDS = (FREE_DATA_DELAY_MINUTES + 1) * 60
POST_CLOSE_POLL_SECONDS = 60


def _market_time_on_date(ny_dt: datetime, minute_of_day: int) -> datetime:
    """Return a New York timestamp on the same date at the requested minute of day."""
    return ny_dt.replace(
        hour=minute_of_day // 60,
        minute=minute_of_day % 60,
        second=0,
        microsecond=0,
    )


def _next_regular_market_open(ny_now: datetime) -> datetime:
    """Return the next regular weekday US equity open in New York time."""
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
    """Return the regular weekday US market session state."""
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
    }


def sleep_until_next_market_open(session: dict) -> None:
    """Sleep while the market is closed so the collector does not poll yfinance overnight."""
    now_ny = session.get("now_ny")
    if not isinstance(now_ny, datetime):
        now_ny = datetime.now(timezone.utc).astimezone(MARKET_TZ)
    next_open = session.get("next_open_dt")
    if not isinstance(next_open, datetime):
        next_open = _next_regular_market_open(now_ny)

    seconds_to_open = max(0.0, (next_open - now_ny).total_seconds())
    next_open_il = next_open.astimezone(LOCAL_TZ)
    print(
        "US market closed. Sleeping until next open: "
        f"NY {next_open.strftime('%Y-%m-%d %H:%M')} | "
        f"Israel {next_open_il.strftime('%Y-%m-%d %H:%M')} "
        f"(~{_format_sleep_duration(seconds_to_open)})."
    )
    time.sleep(max(1.0, seconds_to_open))


def download_yfinance(*args, **kwargs) -> pd.DataFrame:
    """
    Import yfinance only when data is requested.

    This keeps the module importable in test environments before dependencies
    are installed, while still failing clearly when the collector is actually run.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required. Install dependencies with: pip install -r requirements.txt") from exc
    return yf.download(*args, **kwargs)


def _yf_to_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output into the OHLCV schema used by the bots."""
    if data is None or data.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


    # yfinance may return a MultiIndex when symbols are requested in a grouped shape.
    # Flattening keeps the downstream column normalization simple.
    if isinstance(data.columns, pd.MultiIndex):
        flat_cols = []
        for col in data.columns:

            if isinstance(col, tuple):

                flat_cols.append(str(col[0]))
            else:
                flat_cols.append(str(col))
        data = data.copy()
        data.columns = flat_cols

    df = data.reset_index()


    # yfinance changes the timestamp column name depending on interval and source.
    # Pick the best available column and rename it to the bot's standard name.
    if "Datetime" in df.columns:
        ts_col = "Datetime"
    elif "Date" in df.columns:
        ts_col = "Date"
    elif "index" in df.columns:
        ts_col = "index"
    else:

        ts_col = df.columns[0]

    df = df.rename(columns={
        ts_col: "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })


    # Keep only the schema consumed by tradeBot_main.py and tradeBot_Pattern_Test.py.
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


    needed_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df[existing_cols]


    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)


    if "timestamp" in df.columns and isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        # Ensure naive timestamps interpreted as UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)

    # Drop any future timestamps (in case system clock/timezone issues)
    now_utc = datetime.now(timezone.utc)
    future_cutoff = (now_utc + pd.Timedelta(minutes=5)).replace(tzinfo=None)
    before_drop = len(df)
    df = df[df["timestamp"] <= future_cutoff]
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"WARN Dropped {dropped} future bars (> {future_cutoff.isoformat()}) - check system clock/timezone.")

    return df

def fetch_candles_from_start_of_day(min_candles: int = MIN_CANDLES) -> pd.DataFrame:
    """Fetch enough recent candles to seed shared_candles.csv before live polling starts."""
    data = download_yfinance(
        SYMBOL,
        interval=YF_INTERVAL,
        period="5d",
        auto_adjust=False,
        progress=False,
    )

    df = _yf_to_dataframe(data)

    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


    ref_ts = None
    try:
        ref_ts = pd.to_datetime(df["timestamp"]).max()
        if pd.isna(ref_ts):
            ref_ts = None
    except Exception:
        ref_ts = None

    if ref_ts is None:
        ref_ts = datetime.now(timezone.utc)
    else:
        if ref_ts.tzinfo is None:
            ref_ts = ref_ts.replace(tzinfo=timezone.utc)
        else:
            ref_ts = ref_ts.tz_convert(timezone.utc)

    start_of_day = ref_ts.replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
    df_today = df[df["timestamp"] >= start_of_day]

    # Prefer the current trading day. If yfinance returns too few rows, fall back
    # to the latest MIN_CANDLES so the bot can still start with context.
    if len(df_today) >= min_candles:
        return df_today.sort_values("timestamp").reset_index(drop=True)


    if len(df) >= min_candles:
        return df.tail(min_candles).sort_values("timestamp").reset_index(drop=True)


    return df.sort_values("timestamp").reset_index(drop=True)


def enforce_day_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the current day when possible, otherwise keep the latest session-sized window."""
    if df.empty:
        return df


    if "timestamp" not in df.columns:
        print("WARN enforce_day_window: 'timestamp' column not found, returning empty frame.")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")


    if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    ref_ts = None
    try:
        ref_ts = pd.to_datetime(df["timestamp"]).max()
        if pd.isna(ref_ts):
            ref_ts = None
    except Exception:
        ref_ts = None

    if ref_ts is None:
        ref_ts = datetime.now(timezone.utc)
    else:
        if ref_ts.tzinfo is None:
            ref_ts = ref_ts.replace(tzinfo=timezone.utc)
        else:
            ref_ts = ref_ts.tz_convert(timezone.utc)

    start_of_day = ref_ts.replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
    df_today = df[df["timestamp"] >= start_of_day]

    if len(df_today) >= MIN_CANDLES:
        return df_today.reset_index(drop=True)


    tail_count = min(len(df), MIN_CANDLES)
    return df.tail(tail_count).reset_index(drop=True)


COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

df_existing = pd.DataFrame(columns=COLUMNS)

def initialize_candle_file() -> pd.DataFrame:
    """
    Seed shared_candles.csv when the collector is run as a script.

    This intentionally does not run at import time. Importing a module should not
    start network requests or write files, especially in a public portfolio repo.
    """
    try:
        print(f"Fetching candles for {SYMBOL} from yfinance ({INTERVAL_TYPE})...")
        seeded = fetch_candles_from_start_of_day(min_candles=MIN_CANDLES)
        seeded = enforce_day_window(seeded)
        seeded.to_csv(OUTPUT_FILE, index=False)
        print(f"Loaded {len(seeded)} candles into {OUTPUT_FILE}")
        return seeded
    except Exception as e:
        print(f"WARN Failed to fetch start-of-day candles from yfinance: {e}")
        if OUTPUT_FILE.exists():
            try:
                df_raw = pd.read_csv(OUTPUT_FILE)
                if "timestamp" in df_raw.columns:
                    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
                    fallback = enforce_day_window(df_raw)
                    fallback.to_csv(OUTPUT_FILE, index=False)
                    print(f"WARN Falling back to existing file ({len(fallback)} candles)")
                    return fallback
                print("WARN Existing CSV has no 'timestamp' column. Starting fresh.")
            except Exception as e2:
                print(f"ERROR while reading existing CSV: {e2}")

        empty = pd.DataFrame(columns=COLUMNS)
        empty.to_csv(OUTPUT_FILE, index=False)
        return empty


def fetch_latest_candle() -> pd.DataFrame | None:
    """Fetch the newest candle from yfinance; returns None when the feed is unavailable."""
    try:
        data = download_yfinance(
            SYMBOL,
            interval=YF_INTERVAL,
            period="2d",
            auto_adjust=False,
            progress=False,
        )
        df = _yf_to_dataframe(data)
        if df.empty or "timestamp" not in df.columns:
            return None

        latest = df.tail(1)
        return latest.reset_index(drop=True)
    except Exception as e:
        print(f"ERROR fetching latest candle from yfinance: {e}")
        return None


def save_new_candles(new_candles: pd.DataFrame):
    """Merge new candles, remove duplicates, keep the rolling day window, and write the CSV."""
    global df_existing
    combined = pd.concat([df_existing, new_candles], ignore_index=True)
    combined = enforce_day_window(combined)
    combined.to_csv(OUTPUT_FILE, index=False)
    df_existing = combined


def main_loop():
    """Continuously refresh shared_candles.csv for the trading bots."""
    global df_existing
    print(f"Collecting {INTERVAL_TYPE} candles for {SYMBOL} from yfinance (poll {POLL_INTERVAL_SECONDS}s, update on new candle)...")
    last_no_candle_log = 0.0
    initialized_for_active_window = False

    while True:
        session = market_session_status()
        if not session.get("is_active_window"):
            initialized_for_active_window = False
            sleep_until_next_market_open(session)
            continue

        if not initialized_for_active_window:
            df_existing = initialize_candle_file()
            initialized_for_active_window = True

        new_candle = fetch_latest_candle()

        if new_candle is not None and not new_candle.empty:

            if not df_existing.empty and "timestamp" in df_existing.columns:
                last_time = pd.to_datetime(df_existing["timestamp"]).max()
            else:
                last_time = None


            latest_ts = pd.to_datetime(new_candle["timestamp"].iloc[-1])


            # Write only if yfinance advanced to a newer timestamp.
            if last_time is None or pd.isna(last_time) or latest_ts > last_time:
                save_new_candles(new_candles=new_candle)
                print(f"New candle [{latest_ts}] Close: {new_candle.iloc[-1]['close']}")
            else:
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                age_seconds = (now_utc - latest_ts).total_seconds()
                warn_after_seconds = INTERVAL_SECONDS * NO_CANDLE_WARN_MULTIPLIER
                if age_seconds >= warn_after_seconds:
                    now = time.time()
                    if now - last_no_candle_log >= NO_CANDLE_LOG_THROTTLE_SECONDS:
                        age_min = age_seconds / 60.0
                        print(f"Waiting: No new candle (latest {age_min:.1f}m ago).")
                        last_no_candle_log = now
        else:
            print("WARN No data fetched from yfinance.")

        sleep_seconds = (
            POST_CLOSE_POLL_SECONDS
            if session.get("is_post_close_grace") and not session.get("is_open")
            else POLL_INTERVAL_SECONDS
        )
        time.sleep(max(1, int(sleep_seconds)))


if __name__ == "__main__":
    main_loop()
