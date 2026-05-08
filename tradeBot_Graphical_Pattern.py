"""Graphical pattern detectors for the reduced public strategy set.

Each detector returns a boolean Series aligned with the input candle DataFrame.
The public version intentionally includes only five long and five short patterns.
"""

import pandas as pd
import numpy as np

# All detector functions expect a DataFrame with these columns:
# timestamp, open, high, low, close, volume.
# Each detector returns a boolean Series with True on candles where the pattern
# completed. The bot checks the last candle only when deciding whether to enter.

def aggregate_ohlcv_timeframe(df: pd.DataFrame, minutes: int) -> tuple[pd.DataFrame, list[int]]:
    """
    Aggregate 1m candles into a larger timeframe and return (aggregated_df, base_row_positions).
    base_row_positions maps each aggregated row to the last 1m candle index for alignment.
    """
    if df is None or len(df) < minutes:
        return pd.DataFrame(), []
    minutes = int(minutes)
    if minutes <= 1:
        return df.copy(), list(range(len(df)))

    ts = None
    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        except Exception:
            ts = None

    if ts is None or ts.isna().all():
        base = df.reset_index(drop=True).copy()
        group = base.index // minutes
        agg = base.groupby(group).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        counts = base.groupby(group)["close"].count()
        valid = counts >= minutes
        agg = agg.loc[valid].copy()
        last_pos = base.groupby(group).apply(lambda g: int(g.index[-1]))
        last_pos = last_pos.loc[valid].tolist()
        agg.reset_index(drop=True, inplace=True)
        return agg, last_pos

    tmp = df.copy()
    tmp["_orig_idx"] = np.arange(len(tmp))
    tmp["_ts"] = ts
    tmp = tmp.dropna(subset=["_ts"]).sort_values("_ts")
    tmp = tmp.set_index("_ts")
    freq = f"{minutes}min"
    grouped = tmp.resample(freq, label="right", closed="right")
    agg = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    agg["volume"] = grouped["volume"].sum(min_count=1)
    count = grouped["close"].count()
    valid = (count >= minutes) & agg[["open", "high", "low", "close"]].notna().all(axis=1)
    agg = agg.loc[valid].copy()
    last_pos = grouped["_orig_idx"].last().loc[valid].tolist()
    agg.reset_index(drop=True, inplace=True)
    return agg, [int(x) for x in last_pos]

def is_bullish_engulfing(df, min_down_candles=2):
    """Detect a bullish engulfing candle after a short prior down move."""
    if len(df) < 7:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(6, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]


        if pd.isna(prev['open']) or pd.isna(prev['close']) or pd.isna(curr['open']) or pd.isna(curr['close']):
            continue

        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])


        if prev['close'] >= prev['open'] or curr['close'] <= curr['open']:
            continue


        if curr['open'] > prev['close'] or curr['close'] < prev['open']:
            continue


        if curr_body < prev_body * 1.5:
            continue


        closes = df['close'].iloc[i - 5:i - 1]
        down_count = sum(closes.iloc[j] < closes.iloc[j - 1] for j in range(1, len(closes)))
        if down_count < min_down_candles:
            continue

        signal.iloc[i] = True

    return signal

def is_hammer(df, min_drop_pct=0.03, lookback=20):
    """Detect a hammer near the local lows after a meaningful pullback."""
    if len(df) < lookback:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(lookback, len(df)):
        row = df.iloc[i]
        o, c, h, l = row['open'], row['close'], row['high'], row['low']

        if any(pd.isna([o, c, h, l])):
            continue

        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        total_range = h - l if h > l else 1e-6


        if body / total_range > 0.4:
            continue


        if lower_shadow < body * 2:
            continue


        if upper_shadow > body * 0.5:
            continue


        window = df['close'].iloc[i - lookback:i]
        drop_pct = (window.max() - window.min()) / window.max()
        if drop_pct < min_drop_pct:
            continue


        if c > window.min() * 1.03:
            continue

        signal.iloc[i] = True

    return signal

def is_morning_star(df, min_drop_pct=0.03, lookback=5):
    """Detect a three-candle bullish reversal: red candle, small pause, strong green candle."""
    if len(df) < lookback + 5:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(lookback + 2, len(df)):
        a, b, c = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]


        if any(pd.isna(x) for x in [a['open'], a['close'], a['high'], a['low'],
                                    b['open'], b['close'], b['high'], b['low'],
                                    c['open'], c['close'], c['high'], c['low']]):
            continue


        a_body = abs(a['close'] - a['open'])
        a_range = max(a['high'] - a['low'], 1e-6)
        if not (a['close'] < a['open'] and a_body > 0.6 * a_range):
            continue


        b_body = abs(b['close'] - b['open'])
        b_range = max(b['high'] - b['low'], 1e-6)
        if b_body / b_range > 0.4:
            continue


        midpoint = a['open'] - (a_body / 2)
        if not (c['close'] > c['open'] and c['close'] > midpoint):
            continue


        prior = df['close'].iloc[i - lookback - 2:i - 2]
        drop = (prior.max() - prior.min()) / prior.max()
        if drop < min_drop_pct or a['close'] > prior.min() * (1 + min_drop_pct):
            continue

        signal.iloc[i] = True

    return signal

def is_piercing_line(df, min_drop_pct=0.03, lookback=5):
    """Detect a bullish piercing line after a recent decline."""
    if len(df) < lookback + 2:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(lookback + 1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]


        if any(pd.isna(x) for x in [prev['open'], prev['close'], prev['high'], prev['low'],
                                    curr['open'], curr['close'], curr['high'], curr['low']]):
            continue


        prev_body = abs(prev['close'] - prev['open'])
        prev_range = max(prev['high'] - prev['low'], 1e-6)
        if not (prev['close'] < prev['open'] and prev_body > 0.6 * prev_range):
            continue


        midpoint = prev['open'] - (prev_body / 2)
        if not (
            curr['close'] > curr['open'] and
            curr['open'] < prev['low'] and
            midpoint < curr['close'] < prev['open']
        ):
            continue


        prior = df['close'].iloc[i - lookback - 1:i - 1]
        drop = (prior.max() - prior.min()) / prior.max()
        if drop < min_drop_pct or prev['close'] > prior.min() * (1 + min_drop_pct):
            continue

        signal.iloc[i] = True

    return signal

def is_three_white_soldiers(df):
    """Detect three consecutive strong bullish candles after a pullback."""
    if len(df) < 10:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(6, len(df)):
        a = df.iloc[i - 2]
        b = df.iloc[i - 1]
        c = df.iloc[i]


        if any(pd.isna(x) for x in [a['open'], a['close'], b['open'], b['close'], c['open'], c['close']]):
            continue


        if not (a['close'] > a['open'] and b['close'] > b['open'] and c['close'] > c['open']):
            continue


        if not (a['open'] < b['open'] < a['close']):
            continue
        if not (b['open'] < c['open'] < b['close']):
            continue


        if not (b['close'] > a['close'] * 1.001 and c['close'] > b['close'] * 1.001):
            continue


        prev_window = df['close'].iloc[i - 6:i - 3]
        if prev_window.isna().any():
            continue
        drop_pct = (prev_window.max() - prev_window.min()) / prev_window.max()
        if drop_pct < 0.02 or df['close'].iloc[i - 2] > prev_window.min() * 1.03:
            continue

        signal.iloc[i] = True

    return signal

def is_bearish_engulfing(df):
    """Detect a bearish engulfing candle after several bullish candles."""
    if len(df) < 7:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)
    tolerance = 0.002

    for i in range(6, len(df)):
        try:
            prev = df.iloc[i - 1]
            curr = df.iloc[i]


            if prev['close'] <= prev['open']:
                continue


            if curr['close'] >= curr['open']:
                continue


            if curr['open'] < prev['close'] * (1 - tolerance):
                continue
            if curr['close'] > prev['open'] * (1 + tolerance):
                continue


            body = abs(curr['close'] - curr['open'])
            rng = curr['high'] - curr['low'] if curr['high'] > curr['low'] else 1e-6
            if body < rng * 0.5:
                continue


            trend_window = df.iloc[i - 6:i - 1]
            green_candles = (trend_window['close'] > trend_window['open']).sum()
            if green_candles < 3:
                continue

            signal.iloc[i] = True

        except Exception as e:
            print(f"WARN Error in graphical Bearish Engulfing at index {i}: {e}")
            continue

    return signal

def is_shooting_star(df):
    """Detect a shooting star: small body, long upper wick, and prior upward pressure."""
    if len(df) < 6:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(5, len(df)):
        try:
            c = df.iloc[i]
            high = c['high']
            low = c['low']
            open_ = c['open']
            close = c['close']
            body = abs(close - open_)
            rng = high - low if high > low else 1e-6

            upper_shadow = high - max(close, open_)
            lower_shadow = min(close, open_) - low


            if body / rng > 0.4:
                continue


            if upper_shadow < body * 2.5:
                continue


            if lower_shadow > body * 0.3:
                continue


            trend_window = df.iloc[i - 5:i]
            green_count = (trend_window['close'] > trend_window['open']).sum()
            if green_count < 4:
                continue

            signal.iloc[i] = True

        except Exception as e:
            print(f"WARN Error in graphical Shooting Star at index {i}: {e}")
            continue

    return signal

def is_evening_star(df):
    """Detect a three-candle bearish reversal after a short uptrend."""
    if len(df) < 6:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(5, len(df)):
        try:
            n1 = df.iloc[i - 2]
            n2 = df.iloc[i - 1]
            n3 = df.iloc[i]


            trend_window = df.iloc[i - 5:i - 2]
            green_count = (trend_window['close'] > trend_window['open']).sum()
            if green_count < 3:
                continue


            n1_body = abs(n1['close'] - n1['open'])
            n1_range = n1['high'] - n1['low'] if n1['high'] > n1['low'] else 1e-6
            if n1['close'] <= n1['open'] or n1_body < 0.5 * n1_range:
                continue


            n2_body = abs(n2['close'] - n2['open'])
            if n2_body > 0.5 * n1_body:
                continue


            if n3['close'] >= n3['open']:
                continue
            n1_mid = n1['open'] + 0.5 * n1_body
            if n3['close'] > n1_mid:
                continue

            signal.iloc[i] = True

        except Exception as e:
            print(f"WARN Error in graphical Evening Star at index {i}: {e}")
            continue

    return signal

def is_three_black_crows(df):
    """Detect three consecutive bearish candles that step lower."""
    if len(df) < 3:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(2, len(df)):
        try:
            n1 = df.iloc[i - 2]
            n2 = df.iloc[i - 1]
            n3 = df.iloc[i]


            if not all(n['close'] < n['open'] for n in [n1, n2, n3]):
                continue


            if not (n2['open'] < n1['open'] and n2['open'] > n1['close']):
                continue
            if not (n3['open'] < n2['open'] and n3['open'] > n2['close']):
                continue


            if not (n2['close'] < n1['close'] and n3['close'] < n2['close']):
                continue


            valid_wicks = True
            for n in [n1, n2, n3]:
                body = abs(n['close'] - n['open'])
                full_range = n['high'] - n['low'] if n['high'] > n['low'] else 1e-6
                lower_wick = min(n['open'], n['close']) - n['low']
                if body == 0 or lower_wick > 0.5 * body:
                    valid_wicks = False
                    break
            if not valid_wicks:
                continue

            signal.iloc[i] = True

        except Exception as e:
            print(f"WARN Error in graphical Three Black Crows at index {i}: {e}")
            continue

    return signal

def is_dark_cloud_cover(df):
    """Detect a bearish dark-cloud-cover reversal after a strong green candle."""
    if len(df) < 2:
        return pd.Series([False] * len(df), index=df.index)

    signal = pd.Series([False] * len(df), index=df.index)

    for i in range(1, len(df)):
        try:
            prev = df.iloc[i - 1]
            curr = df.iloc[i]


            prev_body = prev['close'] - prev['open']
            prev_range = prev['high'] - prev['low']
            if prev['close'] <= prev['open'] or prev_body < 0.4 * prev_range:
                continue


            if curr['open'] <= prev['high']:
                continue


            if curr['close'] >= curr['open']:
                continue


            mid_prev = prev['open'] + prev_body / 2
            if curr['close'] > mid_prev:
                continue

            signal.iloc[i] = True

        except Exception as e:
            print(f"WARN Error in graphical Dark Cloud Cover at index {i}: {e}")
            continue

    return signal

# Public LONG detector registry. The names must match pattern_list and
# tradeBot_indicators_pattern.confirmation_functions.
LONG_PATTERN_FUNCS = [
    ("bullish_engulfing", is_bullish_engulfing),
    ("hammer", is_hammer),
    ("morning_star", is_morning_star),
    ("piercing_pattern", is_piercing_line),
    ("three_white_soldiers_simple", is_three_white_soldiers),
]

# Public SHORT detector registry. Keep this list aligned with the README.
SHORT_PATTERN_FUNCS = [
    ("bearish_engulfing", is_bearish_engulfing),
    ("shooting_star", is_shooting_star),
    ("evening_star", is_evening_star),
    ("three_black_crows", is_three_black_crows),
    ("dark_cloud_cover", is_dark_cloud_cover),
]

def detect_long_patterns(df):
    """Run all public long-pattern detectors and return {pattern_name: signal_series}."""
    return {name: func(df) for name, func in LONG_PATTERN_FUNCS}


def detect_short_patterns(df):
    """Run all public short-pattern detectors and return {pattern_name: signal_series}."""
    return {name: func(df) for name, func in SHORT_PATTERN_FUNCS}


def detect_Graphical_patterns(df):
    """Run the full reduced public graphical detector set."""
    patterns = {}
    patterns.update(detect_long_patterns(df))
    patterns.update(detect_short_patterns(df))
    return patterns
