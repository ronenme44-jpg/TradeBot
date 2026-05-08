"""Indicator confirmation functions for the selected public pattern set.

Each confirmation function returns a structured result with an overall decision
and a named ``checks`` dictionary that matches ``tradeBot_Indicator_Rules.py``.
"""

import pandas as pd
import numpy as np

# The graphical detector answers "did this candle shape appear?"
# The confirmation functions below answer "did the surrounding indicators support it?"
# Each function returns {"overall": bool, "checks": {...}} so the bot can print
# a readable breakdown and build a reinforcement key from the individual checks.


def _rolling_mean(series, window: int) -> float:
    """Return the latest rolling mean value for a numeric Series."""
    numeric = pd.to_numeric(series, errors='coerce')
    if len(numeric) == 0:
        return float('nan')
    return float(numeric.rolling(window=window, min_periods=max(1, min(window, len(numeric)))).mean().iloc[-1])


def _volatility_estimate(df, idx: int, window: int = 20) -> float:
    """Estimate close-to-close volatility when the caller did not precompute it."""
    if 'close' not in df.columns or idx <= 0:
        return float('nan')
    closes = pd.to_numeric(df['close'].iloc[max(0, idx - window):idx + 1], errors='coerce')
    returns = closes.pct_change().dropna()
    if len(returns) == 0:
        return float('nan')
    return float(returns.std())


def confirm_bullish_engulfing(df, idx):
    """
    Confirm Bullish Engulfing with per-indicator breakdown.
    Returns: {"overall": bool, "checks": {"rsi": bool, "volume": bool, "volatility": bool, "trend": bool}}
    """
    try:
        # --- rsi check ---
        rsi_ok = bool('rsi_14' in df.columns and df['rsi_14'].iloc[idx] <= 45)

        # --- volume check (vs short MA) ---
        vol_now = float(df['volume'].iloc[idx]) if 'volume' in df.columns else float('nan')
        vol_ma = _rolling_mean(df['volume'].iloc[:idx+1], window=10) if 'volume' in df.columns else float('nan')
        volume_ok = bool(vol_now >= 1.1 * vol_ma) if not np.isnan(vol_ma) else False

        # --- volatility check ---
        vol_proxy = float(df['volatility'].iloc[idx]) if 'volatility' in df.columns else _volatility_estimate(df, idx, 20)
        volatility_ok = bool(vol_proxy >= 0.005)

        # --- prior downtrend (last 5 closes decreasing) ---
        if idx < 6 or 'close' not in df.columns:
            trend_ok = False
        else:
            closes = pd.to_numeric(df['close'].iloc[idx-5:idx], errors='coerce')
            trend_ok = bool((closes.diff().dropna() < 0).all())

        checks = {
            'rsi': rsi_ok,
            'volume': volume_ok,
            'volatility': volatility_ok,
            'trend': trend_ok,
        }
        overall = all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_bullish_engulfing: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_hammer(df, idx):
    """
    Confirm Hammer with per-indicator breakdown.
    Returns: {"overall": bool, "checks": {"trend": bool, "rsi": bool, "volume": bool}}
    """
    try:
        if idx < 6 or 'close' not in df.columns:
            return { 'overall': None, 'checks': {} }

        # --- shape gating (hammer body/shadow) ---
        o = float(df.iloc[idx]['open']); c = float(df.iloc[idx]['close'])
        h = float(df.iloc[idx]['high']); l = float(df.iloc[idx]['low'])
        body = abs(c - o)
        total = max(h - l, 1e-9)
        lower_shadow = min(o, c) - l
        shape_ok = (lower_shadow >= 2.0 * body) and (body / total <= 0.30)

        # --- checks aligned with pattern_indicator_rules ---
        closes = pd.to_numeric(df['close'].iloc[max(0, idx-5):idx], errors='coerce')
        trend_ok = bool((closes.diff().dropna() < 0).all() or (len(closes) >= 1 and closes.iloc[-1] < closes.iloc[0]))

        rsi_ok = bool('rsi_14' in df.columns and df['rsi_14'].iloc[idx] <= 45)

        volume_ok = False
        if 'volume' in df.columns:
            vol_ma = _rolling_mean(df['volume'].iloc[:idx+1], window=10)
            if not np.isnan(vol_ma):
                volume_ok = float(df['volume'].iloc[idx]) >= 1.1 * vol_ma

        checks = { 'trend': trend_ok, 'rsi': rsi_ok, 'volume': bool(volume_ok) }
        overall = shape_ok and all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_hammer: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_morning_star(df, idx):
    """
    Confirm Morning Star with per-indicator breakdown.
    Returns: {"overall": bool, "checks": {"trend": bool, "rsi": bool, "volume": bool, "volatility": bool}}
    """
    try:
        if idx < 2:
            return { 'overall': None, 'checks': {} }

        a = df.iloc[idx-2]; b = df.iloc[idx-1]; c = df.iloc[idx]
        if any(pd.isna(x) for x in [a['open'], a['close'], a['high'], a['low'],
                                    b['open'], b['close'], b['high'], b['low'],
                                    c['open'], c['close'], c['high'], c['low']]):
            return { 'overall': None, 'checks': {} }

        # --- shape gating: big red, small middle, strong green above half of first body + gap down ---
        a_body = abs(a['close'] - a['open']); a_range = max(a['high'] - a['low'], 1e-9)
        red_big = (a['close'] < a['open']) and (a_body >= 0.6 * a_range)
        b_body = abs(b['close'] - b['open']); b_range = max(b['high'] - b['low'], 1e-9)
        small_middle = (b_body / b_range) <= 0.4
        midpoint = a['open'] - a_body/2.0
        green_third = (c['close'] > c['open']) and (c['close'] > midpoint)
        gap_down = bool(b['open'] < a['close'])
        shape_ok = red_big and small_middle and green_third and gap_down

        # --- checks aligned with pattern_indicator_rules ---
        start = max(0, idx-2-5)
        prior = pd.to_numeric(df['close'].iloc[start:idx-1], errors='coerce')
        trend_ok = bool((prior.diff().dropna() < 0).sum() >= 3) if len(prior) >= 5 else False

        rsi_ok = bool('rsi_14' in df.columns and df['rsi_14'].iloc[idx] <= 55)

        volume_ok = False
        if 'volume' in df.columns:
            vol_ma = _rolling_mean(df['volume'].iloc[:idx+1], window=10)
            if not np.isnan(vol_ma):
                volume_ok = float(c['volume']) >= 1.1 * vol_ma

        vol_proxy = float(df['volatility'].iloc[idx]) if 'volatility' in df.columns else _volatility_estimate(df, idx, 20)
        volatility_ok = bool(vol_proxy >= 0.004)

        checks = { 'trend': trend_ok, 'rsi': rsi_ok, 'volume': bool(volume_ok), 'volatility': volatility_ok }
        overall = shape_ok and all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_morning_star: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_piercing_line(df, idx):
    """
    Confirm Piercing Line with per-indicator breakdown.
    Returns: {"overall": bool, "checks": {"trend": bool, "rsi": bool, "volume": bool, "volatility": bool}}
    """
    try:
        if idx < 6:
            return { 'overall': None, 'checks': {} }

        # prior downtrend over last 5 bars
        lookback = 5
        close_series = pd.to_numeric(df['close'].iloc[idx - lookback: idx], errors='coerce')
        trend_ok = bool(not close_series.isna().any() and (close_series.diff().dropna() < 0).all())

        rsi_ok = bool('rsi_14' in df.columns and df.loc[idx, 'rsi_14'] <= 45)

        volume_ok = False
        if 'volume' in df.columns:
            mean_vol = float(pd.to_numeric(df['volume'], errors='coerce').rolling(window=5, min_periods=5).mean().iloc[idx])
            if not np.isnan(mean_vol):
                volume_ok = float(df.loc[idx, 'volume']) >= mean_vol

        vol_proxy = float(df.loc[idx, 'volatility']) if 'volatility' in df.columns else _volatility_estimate(df, idx, 20)
        volatility_ok = bool(vol_proxy >= 0.005)

        checks = { 'trend': trend_ok, 'rsi': rsi_ok, 'volume': bool(volume_ok), 'volatility': volatility_ok }
        overall = all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_piercing_line: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_three_white_soldiers(df, idx):
    """
    Confirm Three White Soldiers (simple) with per-indicator breakdown.
    Returns: {"overall": bool, "checks": {"trend": bool, "rsi": bool, "ema_crossover": bool}}
    """
    try:
        if idx < 4 or 'rsi_14' not in df.columns:
            return { 'overall': None, 'checks': {} }

        # prior trend not strongly bullish (flat/down over 3 bars ending at idx-2)
        pre_trend_close = pd.to_numeric(df['close'].iloc[idx - 5:idx - 2], errors='coerce')
        trend_ok = bool(not pre_trend_close.isna().any() and (pre_trend_close.iloc[-1] - pre_trend_close.iloc[0] <= 0))

        # RSI rising last bar vs previous
        rsi_now = df.loc[idx, 'rsi_14']; rsi_prev = df.loc[idx - 1, 'rsi_14']
        rsi_ok = bool(not pd.isna(rsi_now) and not pd.isna(rsi_prev) and rsi_now > rsi_prev)

        ema_crossover_ok = bool(('ema_9' in df.columns) and ('ema_21' in df.columns) and not pd.isna(df.loc[idx, 'ema_9']) and not pd.isna(df.loc[idx, 'ema_21']) and (df.loc[idx, 'ema_9'] > df.loc[idx, 'ema_21']))

        checks = { 'trend': trend_ok, 'rsi': rsi_ok, 'ema_crossover': ema_crossover_ok }
        overall = all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_three_white_soldiers: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_bearish_engulfing(df, idx):
    """
    Confirm Bearish Engulfing with per-indicator breakdown.
    Returns: {"overall": bool, "checks": {"bearish_engulfing": bool, "prior_uptrend": bool, "rsi": bool, "volume_spike": bool, "high_volatility": bool}}
    """
    try:
        if idx < 6:
            return { 'overall': None, 'checks': {} }

        # --- shape gating: current red candle engulfs previous body ---
        prev_open = float(df.loc[idx - 1, 'open'])
        prev_close = float(df.loc[idx - 1, 'close'])
        curr_open = float(df.loc[idx, 'open'])
        curr_close = float(df.loc[idx, 'close'])
        red_now = curr_close < curr_open
        engulfs = (curr_open >= max(prev_open, prev_close)) and (curr_close <= min(prev_open, prev_close))
        shape_ok = red_now and engulfs
        bearish_engulfing_ok = bool(shape_ok)

        # --- trend up in the 5 candles before the previous one ---
        prior = df.iloc[idx - 6: idx - 1]
        trend_ok = bool((prior['close'] > prior['open']).sum() >= 4)

        # --- RSI high on current candle ---
        rsi_ok = bool('rsi_14' in df.columns and df.loc[idx, 'rsi_14'] >= 55)
        volume_spike_ok = False
        if 'volume' in df.columns:
            avg_volume = pd.to_numeric(df['volume'], errors='coerce').iloc[idx - 5: idx].mean()
            if not np.isnan(avg_volume):
                volume_spike_ok = float(df.loc[idx, 'volume']) >= 1.2 * float(avg_volume)

        high_volatility_ok = False
        if 'volatility' in df.columns:
            avg_volatility = pd.to_numeric(df['volatility'], errors='coerce').iloc[idx - 5: idx].mean()
            if not np.isnan(avg_volatility):
                high_volatility_ok = float(df.loc[idx, 'volatility']) >= float(avg_volatility)

        checks = { 'bearish_engulfing': bearish_engulfing_ok, 'prior_uptrend': trend_ok, 'rsi': rsi_ok, 'volume_spike': bool(volume_spike_ok), 'high_volatility': bool(high_volatility_ok) }
        overall = all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_bearish_engulfing: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_shooting_star(df, idx):
    """
    Confirm Shooting Star (bearish) with per-indicator breakdown.
    Returns: {"overall": bool, "checks": {"shooting_star_shape": bool, "prior_uptrend": bool, "rsi": bool, "high_volatility": bool}}
    """
    try:
        if idx < 6:
            return { 'overall': None, 'checks': {} }

        o = float(df.loc[idx, 'open'])
        c = float(df.loc[idx, 'close'])
        h = float(df.loc[idx, 'high'])
        l = float(df.loc[idx, 'low'])

        body = abs(c - o)
        total = max(h - l, 1e-9)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        long_upper_shadow = bool(upper_shadow >= 2.0 * body)
        small_body        = bool(body / total <= 0.30)
        shooting_star_shape_ok = bool(long_upper_shadow and small_body)

        closes = pd.to_numeric(df['close'].iloc[max(0, idx-5):idx], errors='coerce')
        prior_uptrend_ok = bool((closes > pd.to_numeric(df['open'].iloc[max(0, idx-5):idx], errors='coerce')).sum() >= 4)

        rsi_ok = bool('rsi_14' in df.columns and df.loc[idx, 'rsi_14'] >= 60)

        high_volatility_ok = False
        if 'volatility' in df.columns:
            avg_vol = pd.to_numeric(df['volatility'], errors='coerce').iloc[idx - 5:idx].mean()
            if not np.isnan(avg_vol):
                high_volatility_ok = float(df.loc[idx, 'volatility']) >= float(avg_vol)

        checks = {
            'shooting_star_shape': shooting_star_shape_ok,
            'prior_uptrend': prior_uptrend_ok,
            'rsi': rsi_ok,
            'high_volatility': bool(high_volatility_ok),
        }
        overall = all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_shooting_star: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_evening_star(df, idx):
    """
    Confirm Evening Star with per-indicator breakdown.
    Returns: {"overall": bool, "checks": {"prior_uptrend": bool, "strong_first_bullish": bool, "small_body_second": bool, "strong_third_bearish": bool, "rsi": bool, "high_volatility": bool}}
    """
    try:
        if idx < 7:
            return { 'overall': None, 'checks': {} }

        # candle refs
        o1, c1 = df.loc[idx - 2, 'open'], df.loc[idx - 2, 'close']
        o2, c2 = df.loc[idx - 1, 'open'], df.loc[idx - 1, 'close']
        o3, c3 = df.loc[idx,     'open'], df.loc[idx,     'close']

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        direction1 = c1 > o1
        direction3 = c3 < o3

        # shape components
        strong_first_bullish_ok = bool(direction1 and body1 >= 0.005)
        small_body_second_ok   = bool(body2 <= 0.5 * body1)
        strong_third_bearish_ok = bool(direction3 and (c3 <= (o1 + body1 / 2.0)))

        # prior uptrend: at least 3 of 5 greens before the pattern's first candle
        up_count = int((df.loc[idx - 7:idx - 3, 'close'] > df.loc[idx - 7:idx - 3, 'open']).sum())
        prior_uptrend_ok = bool(up_count >= 3)

        rsi_ok = bool('rsi_14' in df.columns and df.loc[idx, 'rsi_14'] >= 60)

        high_volatility_ok = False
        if 'volatility' in df.columns:
            avg_volatility = pd.to_numeric(df['volatility'], errors='coerce').iloc[idx - 5:idx].mean()
            if not np.isnan(avg_volatility):
                high_volatility_ok = float(df.loc[idx, 'volatility']) >= float(avg_volatility)

        checks = {
            'prior_uptrend': prior_uptrend_ok,
            'strong_first_bullish': strong_first_bullish_ok,
            'small_body_second': small_body_second_ok,
            'strong_third_bearish': strong_third_bearish_ok,
            'rsi': rsi_ok,
            'high_volatility': bool(high_volatility_ok)
        }
        overall = all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_evening_star: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_three_black_crows(df, idx):
    """
    Confirm Three Black Crows with per-indicator breakdown.
    Returns checks aligned with tradeBot_Indicator_Rules.py:
    three_red_candles, inside_previous_body, short_lower_wick, rsi_condition,
    high_volatility.
    """
    try:
        if idx < 5:
            return { 'overall': None, 'checks': {} }

        # Shape checks are kept separate because each one becomes part of the
        # reinforcement key. This makes the score file explain which condition
        # helped or hurt a pattern over time.
        three_red_candles_ok = True
        inside_previous_body_ok = True
        short_lower_wick_ok = True
        for i in range(3):
            c_idx = idx - i
            o = float(df.loc[c_idx, 'open']); c = float(df.loc[c_idx, 'close'])
            h = float(df.loc[c_idx, 'high']); l = float(df.loc[c_idx, 'low'])
            body = abs(o - c)
            lower_wick = min(o, c) - l
            if not (c < o):
                three_red_candles_ok = False
            if body == 0 or (lower_wick / max(body, 1e-9) > 0.5):
                short_lower_wick_ok = False
            if i > 0:
                prev_o = float(df.loc[c_idx + 1, 'open']); prev_c = float(df.loc[c_idx + 1, 'close'])
                body_high = max(prev_o, prev_c); body_low = min(prev_o, prev_c)
                if not (body_low <= o <= body_high):
                    inside_previous_body_ok = False

        # checks
        rsi_or_drop_ok = False
        try:
            rsi_before = float(df.loc[idx - 3, 'rsi_14']) if 'rsi_14' in df.columns else np.nan
        except Exception:
            rsi_before = np.nan
        price_then = float(df.loc[idx - 2, 'close'])
        price_now = float(df.loc[idx, 'close'])
        drop_pct = (price_then - price_now) / max(price_then, 1e-9)
        rsi_or_drop_ok = (not np.isnan(rsi_before) and rsi_before >= 50) or (drop_pct >= 0.02)

        volatility_ok = False
        if 'volatility' in df.columns:
            avg_vol = pd.to_numeric(df['volatility'], errors='coerce').iloc[idx - 5:idx].mean()
            vol_now = float(df.loc[idx, 'volatility'])
            if not np.isnan(avg_vol):
                volatility_ok = vol_now >= max(0.006, float(avg_vol))

        checks = {
            'three_red_candles': bool(three_red_candles_ok),
            'inside_previous_body': bool(inside_previous_body_ok),
            'short_lower_wick': bool(short_lower_wick_ok),
            'rsi_condition': bool(rsi_or_drop_ok),
            'high_volatility': bool(volatility_ok),
        }
        overall = all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_three_black_crows: {e}")
        return { 'overall': None, 'checks': {} }

def confirm_dark_cloud_cover(df, idx):
    """
    Confirm Dark Cloud Cover with per-indicator breakdown.
    Returns checks aligned with tradeBot_Indicator_Rules.py:
    prev_green_strong, gap_up_open, close_below_midpoint, rsi_condition,
    vol_or_volatility_high.
    """
    try:
        if idx < 5:
            return { 'overall': None, 'checks': {} }

        prev_open = float(df.loc[idx - 1, 'open']); prev_close = float(df.loc[idx - 1, 'close'])
        prev_high = float(df.loc[idx - 1, 'high'])
        prev_body = abs(prev_close - prev_open)

        curr_open = float(df.loc[idx, 'open']); curr_close = float(df.loc[idx, 'close'])

        # shape gating: prior strong green, gap up, close below midpoint of prior body
        shape_ok = (
            (prev_close > prev_open) and (prev_body >= 0.001) and
            (curr_open > prev_high) and
            (curr_close < (prev_open + (prev_close - prev_open) / 2.0))
        )

        # checks
        trend_slice = pd.to_numeric(df.loc[idx - 5:idx - 1, 'close'], errors='coerce')
        trend_ok = bool(len(trend_slice) == 5 and (trend_slice.diff() > 0).sum() >= 3)

        rsi_ok = bool('rsi_14' in df.columns and df.loc[idx, 'rsi_14'] >= 55)

        vol_ok = False
        if 'volume' in df.columns:
            avg_vol = pd.to_numeric(df['volume'], errors='coerce').iloc[idx - 5:idx].mean()
            if not np.isnan(avg_vol):
                vol_ok = float(df.loc[idx, 'volume']) >= 1.1 * float(avg_vol)

        volat_ok = False
        if 'volatility' in df.columns:
            avg_volat = pd.to_numeric(df['volatility'], errors='coerce').iloc[idx - 5:idx].mean()
            if not np.isnan(avg_volat):
                volat_ok = float(df.loc[idx, 'volatility']) >= 1.2 * float(avg_volat)

        activity_ok = bool(vol_ok or volat_ok)

        checks = {
            'prev_green_strong': bool(prev_close > prev_open and prev_body >= 0.001),
            'gap_up_open': bool(curr_open > prev_high),
            'close_below_midpoint': bool(curr_close < (prev_open + (prev_close - prev_open) / 2.0)),
            'rsi_condition': rsi_ok,
            'vol_or_volatility_high': activity_ok,
        }
        overall = shape_ok and all(checks.values())
        return { 'overall': overall, 'checks': checks }

    except Exception as e:
        print(f"ERROR in confirm_dark_cloud_cover: {e}")
        return { 'overall': None, 'checks': {} }

# Public confirmation registry. Keys must match the graphical detector names and
# the pattern_indicator_rules schema.
confirmation_functions = {
    "bullish_engulfing": confirm_bullish_engulfing,
    "hammer": confirm_hammer,
    "morning_star": confirm_morning_star,
    "piercing_pattern": confirm_piercing_line,
    "three_white_soldiers_simple": confirm_three_white_soldiers,
    "bearish_engulfing": confirm_bearish_engulfing,
    "shooting_star": confirm_shooting_star,
    "evening_star": confirm_evening_star,
    "three_black_crows": confirm_three_black_crows,
    "dark_cloud_cover": confirm_dark_cloud_cover,
}
