"""Optional trade-tape adapter utilities.

The main public workflow uses OHLCV candles, but this adapter is retained so the
main bot can also read a JSONL tick/trade feed when one is available.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# TTP schema used by this file:
# Symbol: ticker, Last: last trade price, Volume: trade size, Time: Unix ms.
# The main bot can read a JSONL file of these ticks and convert it to candles.

def normalize_symbol(symbol: str | None) -> str | None:
    """Return a clean, uppercase US ticker (letters/numbers/dot/hyphen only)."""
    if symbol is None:
        return None
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    cleaned = "".join(ch for ch in str(symbol).strip().upper() if ch in allowed)
    return cleaned or None


def iso_to_unix_ms(ts_val) -> int | None:
    """Convert ISO8601 or datetime-like inputs to Unix milliseconds."""
    if ts_val is None:
        return None
    if isinstance(ts_val, (int, float)):
        val = float(ts_val)
        # Treat large numbers as already-ms; otherwise seconds -> ms
        return int(val) if val >= 1e12 else int(val * 1000)
    if isinstance(ts_val, pd.Timestamp):
        ts_val = ts_val.to_pydatetime()
    if isinstance(ts_val, datetime):
        dt = ts_val if ts_val.tzinfo else ts_val.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if isinstance(ts_val, str):
        try:
            dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def unix_ms_to_iso(ms_val: int | float | None) -> str | None:
    """Convert Unix milliseconds to a canonical UTC ISO string."""
    if ms_val is None:
        return None
    try:
        return datetime.fromtimestamp(float(ms_val) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def alpaca_to_ttp(message: dict) -> dict | None:
    """
    Translate an Alpaca trade/quote message into the TTP schema.
    Maps: S->Symbol, p->Last, bp->Bid, ap->Ask, s->Volume, t->Time (Unix ms).
    """
    if not isinstance(message, dict):
        return None

    # Alpaca-style messages use compact keys. Convert them into the neutral
    # schema above so the rest of the bot does not depend on one vendor format.
    msg_type_raw = message.get("T")
    msg_type = str(msg_type_raw).lower() if msg_type_raw is not None else ""
    symbol = normalize_symbol(message.get("S") or message.get("symbol") or message.get("Symbol"))

    ttp: dict = {}
    if symbol:
        ttp["Symbol"] = symbol

    time_val = message.get("t") or message.get("timestamp") or message.get("time")
    time_ms = iso_to_unix_ms(time_val)
    if time_ms is not None:
        ttp["Time"] = int(time_ms)

    if msg_type == "t" or (msg_type == "" and "p" in message):
        price = message.get("p") or message.get("price")
        size = message.get("s") or message.get("size") or message.get("q")
        if price is not None:
            try:
                ttp["Last"] = float(price)
            except Exception:
                pass
        if size is not None:
            try:
                vol = float(size)
                ttp["Volume"] = int(vol) if float(vol).is_integer() else vol
            except Exception:
                pass
    elif msg_type == "q":
        bid = message.get("bp") or message.get("bid_price") or message.get("bid")
        ask = message.get("ap") or message.get("ask_price") or message.get("ask")
        if bid is not None:
            try:
                ttp["Bid"] = float(bid)
            except Exception:
                pass
        if ask is not None:
            try:
                ttp["Ask"] = float(ask)
            except Exception:
                pass

    return ttp or None


def floor_shares(value) -> int:
    """TTP rule: shares must be integers (no fractional sizing)."""
    try:
        return max(0, int(math.floor(float(value))))
    except Exception:
        return 0


class TickCandleBuilder:
    """Aggregate trade ticks into rolling candles without subscribing to bar data."""

    def __init__(self, interval_seconds: int = 60, history_limit: int = 1500, seed_df: pd.DataFrame | None = None):
        # interval_seconds controls the candle size. For the public bot this is
        # usually 60 seconds, matching the 1-minute pattern logic.
        self.interval_seconds = max(1, int(interval_seconds))
        self.history = deque(maxlen=history_limit)
        self.current_bucket_ms: int | None = None
        self.current_candle: dict | None = None
        self.last_tick_ms: int | None = None
        self.last_trade_price: float | None = None
        if seed_df is not None and len(seed_df) > 0:
            self._bootstrap(seed_df)

    def _bootstrap(self, df: pd.DataFrame) -> None:
        """Seed the builder with existing OHLCV candles before live ticks arrive."""
        try:
            seed = df.sort_values("timestamp")
        except Exception:
            seed = df
        for _, row in seed.iterrows():
            try:
                ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                candle = {
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "open": float(row.get("open", np.nan)),
                    "high": float(row.get("high", np.nan)),
                    "low": float(row.get("low", np.nan)),
                    "close": float(row.get("close", np.nan)),
                    "volume": float(row.get("volume", 0.0) or 0.0),
                }
                self.history.append(candle)
                if np.isfinite(candle["close"]):
                    self.last_trade_price = candle["close"]
            except Exception:
                continue

    def _start_candle(self, bucket_ms: int, price: float, tick: dict) -> None:
        """Start a new candle bucket from the first tick in that time window."""
        self.current_bucket_ms = bucket_ms
        ts_iso = unix_ms_to_iso(bucket_ms) or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.current_candle = {
            "timestamp": ts_iso,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": float(tick.get("Volume", 0.0) or 0.0),
        }

    def _finalize_current(self) -> None:
        """Move the in-progress candle into history when its time bucket closes."""
        if self.current_candle:
            self.history.append(self.current_candle)
        self.current_bucket_ms = None
        self.current_candle = None

    def _update_current(self, price: float, tick: dict) -> None:
        """Update high, low, close, and volume for the current candle."""
        if self.current_candle is None:
            self._start_candle(self.current_bucket_ms or int(time.time() * 1000), price, tick)
            return
        self.current_candle["close"] = price
        self.current_candle["high"] = max(self.current_candle.get("high", price), price)
        self.current_candle["low"] = min(self.current_candle.get("low", price), price)
        vol_add = float(tick.get("Volume", 0.0) or 0.0)
        try:
            self.current_candle["volume"] = float(self.current_candle.get("volume", 0.0)) + vol_add
        except Exception:
            self.current_candle["volume"] = vol_add

    def update_with_tick(self, tick: dict) -> tuple[pd.DataFrame | None, bool]:
        """
        Ingest a single TTP-formatted tick and update the in-flight candle.
        Returns (dataframe_with_current, bool_closed_previous_candle).
        """
        if not isinstance(tick, dict):
            return None, False

        price = tick.get("Last")
        try:
            price = float(price)
        except Exception:
            return self.as_dataframe(include_current=True), False

        ts_ms = tick.get("Time")
        if ts_ms is None:
            ts_ms = iso_to_unix_ms(tick.get("t"))
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        ts_ms = int(ts_ms)
        self.last_tick_ms = ts_ms

        interval_ms = self.interval_seconds * 1000
        bucket_ms = int(ts_ms - (ts_ms % interval_ms))

        closed_prev = False
        if self.current_bucket_ms is None:
            self._start_candle(bucket_ms, price, tick)
        elif bucket_ms > self.current_bucket_ms:
            self._finalize_current()
            self._start_candle(bucket_ms, price, tick)
            closed_prev = True
        else:
            self._update_current(price, tick)

        self.last_trade_price = price
        return self.as_dataframe(include_current=True), closed_prev

    def as_dataframe(self, include_current: bool = True) -> pd.DataFrame | None:
        """Return candle history as a DataFrame in the same schema as shared_candles.csv."""
        rows = list(self.history)
        if include_current and self.current_candle:
            rows.append(self.current_candle)
        if not rows:
            return None
        try:
            df = pd.DataFrame(rows)
            df = df.dropna(subset=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame(rows)


def persist_ttp_tick(path: str, tick: dict) -> None:
    """Append a TTP tick to a JSONL file (lightweight shared feed)."""
    if not isinstance(tick, dict):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a") as f:
        f.write(json.dumps(tick) + "\n")


def load_ttp_jsonl_to_df(path: str, interval_seconds: int = 60, limit: int = 1500, symbol: str | None = None) -> pd.DataFrame | None:
    """
    Build candles from a shared JSONL tick feed (TTP schema).
    Useful for multiple bots consuming the same converted stream.
    """
    if not os.path.exists(path):
        return None
    builder = TickCandleBuilder(interval_seconds=interval_seconds, history_limit=limit)
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                ttp_tick = alpaca_to_ttp(raw) if raw and "Last" not in raw else raw
                if not ttp_tick:
                    continue
                # If a symbol is supplied, ignore ticks for other symbols.
                sym = ttp_tick.get("Symbol")
                if symbol and sym and sym != normalize_symbol(symbol):
                    continue
                builder.update_with_tick(ttp_tick)
    except Exception:
        return builder.as_dataframe(include_current=True)
    return builder.as_dataframe(include_current=True)


def iter_trade_ticks(symbol: str, seed_df: pd.DataFrame | None = None):
    """
    Yield raw Alpaca-style trade messages (no bar subscriptions).
    If ALPACA_TRADES_JSONL is set, stream from that file; otherwise use seed_df or a mock feed.
    """
    normalized = normalize_symbol(symbol) or symbol
    jsonl_path = os.environ.get("ALPACA_TRADES_JSONL")

    # Optional integration point: set ALPACA_TRADES_JSONL to replay a saved
    # trade stream without changing the bot code.
    if jsonl_path and os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    if isinstance(msg, dict):
                        msg.setdefault("S", normalized)
                        yield msg
                except Exception:
                    continue
        return

    if seed_df is not None and len(seed_df) > 0:
        # Offline path: turn existing candles into one synthetic trade per candle.
        try:
            seed = seed_df.sort_values("timestamp")
        except Exception:
            seed = seed_df
        for _, row in seed.iterrows():
            ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            yield {
                "T": "t",
                "S": normalized,
                "p": float(row.get("close", row.get("open", 0.0))),
                "s": float(row.get("volume", 0.0) or 0.0),
                "t": ts.isoformat()
            }
        return

    # No live feed was configured and no seed data was supplied.
    # Stop cleanly instead of generating synthetic market data.
    return
