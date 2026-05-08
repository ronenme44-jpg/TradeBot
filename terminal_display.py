"""Compact terminal rendering helpers for bot snapshots.

The helpers in this file keep runtime output readable without coupling the
trading logic to a specific UI framework.
"""

from __future__ import annotations

import pandas as pd


def format_percent(value):
    """Return signed percent text for fractional values."""
    try:
        sign = "+" if value > 0 else "-" if value < 0 else "+/-"
        return f"{sign}{abs(float(value) * 100):.2f}%"
    except Exception:
        return "+/-0.00%"


def _fmt_row(row):
    """Format one candle row for compact console output."""
    try:
        ts = row["timestamp"] if "timestamp" in row else ""
        if isinstance(ts, str) and len(ts) >= 8:
            ts = ts.rstrip("Z")
            ts = ts[-8:]
        time_part = ts[:5] if isinstance(ts, str) and len(ts) >= 5 else ts
        return f"Time {time_part} | close {row['close']:.2f}"
    except Exception:
        return str(row)


def render_status(
    df: pd.DataFrame,
    *,
    open_trade: dict | None,
    checks: dict | None = None,
    pattern_name: str | None = None,
    candles: int = 1,
) -> None:
    """
    A single, centralized snapshot of the last candle, optional checks, and trade status.
    Keep output compact and readable for terminal use.
    """
    try:
        print("---------------- Snapshot ----------------")
        # Print the latest candle context first, because every signal is based
        # on the newest row in the DataFrame.
        if isinstance(df, pd.DataFrame) and len(df) >= 1:
            n = max(1, int(candles)) if isinstance(candles, (int, float)) else 1
            tail = df.tail(n).reset_index(drop=True)
            for i in range(len(tail)):
                print(_fmt_row(tail.iloc[i]))

        if pattern_name:
            print(f"Pattern: {pattern_name}")

        if checks:
            # Indicator checks are shown as pass/fail so a reviewer can see why
            # a pattern was strong or weak without reading the JSON files.
            trues = sum(1 for v in checks.values() if bool(v))
            falses = sum(1 for v in checks.values() if not bool(v))
            parts = ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items())
            print(f"Checks: {trues} passed / {falses} failed -> {parts}")

        if open_trade and open_trade.get("status") == "open":
            # The open-trade line mirrors the fields stored in pending_pattern JSON.
            side = open_trade.get("signal", "LONG")
            entry = open_trade.get("entry_price")
            tp = open_trade.get("tp_price")
            sl = open_trade.get("sl_price")
            lvl = open_trade.get("current_level", 0)
            print(f"Open: {side} @ {entry:.4f} | Lv {lvl} | TP {tp:.4f} | SL {sl:.4f}")

        print("------------------------------------------")
    except Exception as e:
        print(f"[display error] {e}")
