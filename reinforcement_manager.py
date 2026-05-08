"""Persist reinforcement-style scores for pattern outcome keys.

The trading bots build deterministic pattern keys from indicator outcomes.
This module stores aggregate counts, scores, and basic success statistics in JSON.
"""

from __future__ import annotations

import json
import os
from typing import Dict

# Path to the reinforcement scores file (bots override this at runtime).
# tradeBot_main.py and tradeBot_Pattern_Test.py set REINFORCEMENT_PATH after import
# so each bot writes to its own JSON file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REINFORCEMENT_PATH = os.path.join(BASE_DIR, "Bot2_reinforcement_scores.json")

def load_scores() -> Dict:
    """Load reinforcement scores dictionary from JSON file."""
    if not os.path.exists(REINFORCEMENT_PATH):
        return {}
    try:
        with open(REINFORCEMENT_PATH, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"WARN Could not load scores file: {e}")
        return {}


def save_scores(scores: Dict) -> None:
    """Persist reinforcement scores dictionary to JSON file."""
    try:
        with open(REINFORCEMENT_PATH, "w") as f:
            json.dump(scores, f, indent=4)
    except Exception as e:
        print(f"WARN Could not save scores file: {e}")


def ensure_key(scores: Dict, key: str) -> None:
    """Ensure the key exists in the scores dict with default structure."""
    if key not in scores:
        # This schema is intentionally plain JSON so reviewers can inspect scores
        # without any database or external service.
        scores[key] = {
            "score": 0.0,
            "wins": 0,
            "losses": 0,
            "trades": 0,
            "total_profit": 0.0,        # cumulative net profit added to the score
            "avg_profit": None,          # average net profit per trade
            "total_duration_min": 0.0,
            "avg_duration_min": None,
            "last_pnl": None,
            "last_pnl_fraction": None,
            "updated_at": None,
        }


def append_detailed_log(event: dict) -> None:
    """Reserved extension hook for detailed audit logging."""
    return


def get_score(key: str) -> float:
    """Return the numeric score for one reinforcement key, defaulting to 0."""
    scores = load_scores()
    if key in scores:
        try:
            return float(scores[key].get("score", 0))
        except Exception:
            return 0.0
    return 0.0


def update_score_on_close(
    key: str,
    net_pnl_fraction: float | None = None,
    net_profit: float | None = None,
    *,
    trade_duration_min: float | None = None,
    timestamp: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    """
    Update reinforcement statistics after a trade closes.

    A key represents both the pattern name and the indicator outcomes at entry.
    This lets the bot learn that the same graphical pattern may behave
    differently under different indicator conditions.
    """
    scores = load_scores()
    ensure_key(scores, key)

    record = scores[key]
    old_score = float(record.get("score", 0))

    # Keep a running count for averaging duration (handles older files gracefully)
    try:
        record["trades"] = int(record.get("trades", 0))
    except Exception:
        record["trades"] = 0
    record["trades"] += 1

    # Determine value to add to the score (prefers actual net profit dollars)
    pnl_value = 0.0
    if net_profit is not None:
        try:
            pnl_value = float(net_profit)
        except Exception:
            pnl_value = 0.0
    elif net_pnl_fraction is not None:
        try:
            pnl_value = float(net_pnl_fraction)
        except Exception:
            pnl_value = 0.0

    if pnl_value > 0:
        record["wins"] = int(record.get("wins", 0)) + 1
    elif pnl_value < 0:
        record["losses"] = int(record.get("losses", 0)) + 1

    record["score"] = round(old_score + pnl_value, 4)
    record["last_pnl"] = pnl_value
    record["last_pnl_fraction"] = net_pnl_fraction
    record["updated_at"] = timestamp

    # Track cumulative and average net profit per trade
    try:
        record["total_profit"] = round(float(record.get("total_profit", 0.0)) + pnl_value, 4)
        if record["trades"] > 0:
            record["avg_profit"] = round(record["total_profit"] / record["trades"], 4)
    except Exception:
        pass

    # Track duration (in minutes) and keep an average per key
    if trade_duration_min is not None:
        try:
            duration_val = max(0.0, float(trade_duration_min))
            total_duration = float(record.get("total_duration_min", 0.0)) + duration_val
            record["total_duration_min"] = round(total_duration, 2)
            if record["trades"] > 0:
                record["avg_duration_min"] = round(total_duration / record["trades"], 2)
        except Exception:
            pass

    scores[key] = record
    save_scores(scores)

    append_detailed_log({
        "event": "trade_closed",
        "key": key,
        "net_pnl": net_pnl_fraction,
        "net_profit": net_profit,
        "delta": pnl_value,
        "old_score": old_score,
        "new_score": record["score"],
        "wins": record["wins"],
        "losses": record["losses"],
        "trades": record.get("trades"),
        "total_profit": record.get("total_profit"),
        "avg_profit": record.get("avg_profit"),
        "avg_duration_min": record.get("avg_duration_min"),
        "timestamp": timestamp,
        "symbol": symbol,
        "timeframe": timeframe,
    })

    return record
