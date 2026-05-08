"""Indicator rule schema for the selected public pattern set.

The rule list defines the stable order used to build reinforcement keys from
confirmation outcomes. Keeping this schema explicit makes pattern scoring
deterministic and easy to inspect.
"""

from reinforcement_manager import load_scores, save_scores, ensure_key

# Pattern rules define the exact indicator names that should appear in each
# confirmer's "checks" dictionary. The order matters because it creates stable
# reinforcement keys across runs.
pattern_indicator_rules = {
    "bullish_engulfing": [
        {"indicator": "rsi", "expected": True},
        {"indicator": "volume", "expected": True},
        {"indicator": "volatility", "expected": True},
        {"indicator": "trend", "expected": True},
    ],
    "hammer": [
        {"indicator": "trend", "expected": True},
        {"indicator": "rsi", "expected": True},
        {"indicator": "volume", "expected": True},
    ],
    "morning_star": [
        {"indicator": "trend", "expected": True},
        {"indicator": "rsi", "expected": True},
        {"indicator": "volume", "expected": True},
        {"indicator": "volatility", "expected": True},
    ],
    "piercing_pattern": [
        {"indicator": "trend", "expected": True},
        {"indicator": "rsi", "expected": True},
        {"indicator": "volume", "expected": True},
        {"indicator": "volatility", "expected": True},
    ],
    "three_white_soldiers_simple": [
        {"indicator": "trend", "expected": True},
        {"indicator": "rsi", "expected": True},
        {"indicator": "ema_crossover", "expected": True},
    ],
    "bearish_engulfing": [
        {"indicator": "bearish_engulfing", "expected": True},
        {"indicator": "prior_uptrend", "expected": True},
        {"indicator": "rsi", "expected": True},
        {"indicator": "volume_spike", "expected": True},
        {"indicator": "high_volatility", "expected": True},
    ],
    "shooting_star": [
        {"indicator": "shooting_star_shape", "expected": True},
        {"indicator": "prior_uptrend", "expected": True},
        {"indicator": "rsi", "expected": True},
        {"indicator": "high_volatility", "expected": True},
    ],
    "evening_star": [
        {"indicator": "prior_uptrend", "expected": True},
        {"indicator": "strong_first_bullish", "expected": True},
        {"indicator": "small_body_second", "expected": True},
        {"indicator": "strong_third_bearish", "expected": True},
        {"indicator": "rsi", "expected": True},
        {"indicator": "high_volatility", "expected": True},
    ],
    "three_black_crows": [
        {"indicator": "three_red_candles", "expected": True},
        {"indicator": "inside_previous_body", "expected": True},
        {"indicator": "short_lower_wick", "expected": True},
        {"indicator": "rsi_condition", "expected": True},
        {"indicator": "high_volatility", "expected": True},
    ],
    "dark_cloud_cover": [
        {"indicator": "prev_green_strong", "expected": True},
        {"indicator": "gap_up_open", "expected": True},
        {"indicator": "close_below_midpoint", "expected": True},
        {"indicator": "rsi_condition", "expected": True},
        {"indicator": "vol_or_volatility_high", "expected": True},
    ],
}

def build_pattern_key(pattern_name: str, outcomes: dict) -> str:
    """
    Build a deterministic reinforcement key for one pattern and check result.

    Example:
    bullish_engulfing|rsi:1,volume:0,volatility:1,trend:1
    """
    if pattern_name not in pattern_indicator_rules:
        return pattern_name
    pieces = []
    for rule in pattern_indicator_rules[pattern_name]:
        indicator = rule.get("indicator")
        expected = bool(rule.get("expected", True))
        passed = bool(outcomes.get(indicator, False))
        pieces.append(f"{indicator}:{int(passed == expected)}")
    return f"{pattern_name}|" + ",".join(pieces)


def ensure_reinforcement_key(pattern_name: str, outcomes: dict) -> str:
    """Create the key in the score file if it does not already exist."""
    key = build_pattern_key(pattern_name, outcomes)
    scores = load_scores()
    ensure_key(scores, key)
    save_scores(scores)
    return key
