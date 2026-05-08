"""Pattern naming helpers retained for compatibility with the full project.

The public pattern set does not expand to every private timeframe variant, but
these helpers document and preserve the original naming convention.
"""

# The full project can expand a base pattern into 5-minute and 15-minute names.
# The public bot keeps the reduced one-minute list active, but this helper is
# retained so the naming convention is easy to understand.
TIMEFRAME_MINUTES = (5, 15)
TIMEFRAME_SUFFIXES = {5: "5Minutes", 15: "15Minutes"}
NEGATIVE_SUFFIX = "_negative"
EXCLUDE_PREFIXES = ("hourly_", "close_")


def should_timeframe_expand(name: str | None) -> bool:
    """Return True when a base pattern name can be expanded to timeframe variants."""
    if not isinstance(name, str) or not name:
        return False
    if name.endswith(NEGATIVE_SUFFIX):
        return False
    if name.endswith("_5Minutes") or name.endswith("_15Minutes"):
        return False
    return not any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def extend_pattern_list(pattern_list: list[dict], minutes: tuple[int, ...] = TIMEFRAME_MINUTES) -> list[dict]:
    """Return pattern_list plus derived timeframe and inverse-direction variants."""
    if not isinstance(pattern_list, list):
        return pattern_list

    existing = {item.get("name") for item in pattern_list if isinstance(item, dict)}
    additions: list[dict] = []

    for item in pattern_list:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        ptype = str(item.get("type", "LONG")).upper()
        if not should_timeframe_expand(name):
            continue
        for minutes_val in minutes:
            # Example: bullish_engulfing -> bullish_engulfing_5Minutes.
            suffix = TIMEFRAME_SUFFIXES.get(minutes_val, f"{minutes_val}Minutes")
            tf_name = f"{name}_{suffix}"
            if tf_name not in existing:
                additions.append({"name": tf_name, "type": ptype})
                existing.add(tf_name)
            neg_name = f"{tf_name}{NEGATIVE_SUFFIX}"
            # Negative variants flip the side: a LONG base pattern becomes a SHORT variant.
            neg_type = "SHORT" if ptype == "LONG" else "LONG"
            if neg_name not in existing:
                additions.append({"name": neg_name, "type": neg_type})
                existing.add(neg_name)

    return pattern_list + additions
