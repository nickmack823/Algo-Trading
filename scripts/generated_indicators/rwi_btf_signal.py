import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_rwi_btf(df: pd.DataFrame) -> list[str]:
    # Extract series safely
    rwi_h = pd.to_numeric(df.get('RWIH', pd.Series(index=df.index, dtype=float)), errors='coerce')
    rwi_l = pd.to_numeric(df.get('RWIL', pd.Series(index=df.index, dtype=float)), errors='coerce')

    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Get last two values (use NaNs when not available)
    if n >= 2:
        h_prev, h_curr = rwi_h.iloc[-2], rwi_h.iloc[-1]
        l_prev, l_curr = rwi_l.iloc[-2], rwi_l.iloc[-1]
    else:
        h_prev, h_curr = pd.NA, rwi_h.iloc[-1] if len(rwi_h) else pd.NA
        l_prev, l_curr = pd.NA, rwi_l.iloc[-1] if len(rwi_l) else pd.NA

    thr = 1.0

    # Helper checks robust to NaNs
    def valid(x): return pd.notna(x)
    def crossed_up(a, b, level): return valid(a) and valid(b) and a <= level and b > level
    def ge(a, b): return valid(a) and valid(b) and a >= b
    def gt(a, b): return valid(a) and valid(b) and a > b
    def le(a, b): return valid(a) and valid(b) and a <= b

    # Threshold cross conditions (last two bars)
    bull_thr_cross = crossed_up(h_prev, h_curr, thr)
    bear_thr_cross = crossed_up(l_prev, l_curr, thr)

    # Dominance cross conditions (RWIH vs RWIL), confirm with > 1 to filter noise
    bull_dom_cross = valid(h_prev) and valid(l_prev) and valid(h_curr) and valid(l_curr) and le(h_prev, l_prev) and gt(h_curr, l_curr) and gt(h_curr, thr)
    bear_dom_cross = valid(h_prev) and valid(l_prev) and valid(h_curr) and valid(l_curr) and le(l_prev, h_prev) and gt(l_curr, h_curr) and gt(l_curr, thr)

    # Decide signal
    signal = NO_SIGNAL
    if bull_thr_cross and (not valid(l_curr) or ge(h_curr, l_curr)):
        signal = BULLISH_SIGNAL
    elif bear_thr_cross and (not valid(h_curr) or ge(l_curr, h_curr)):
        signal = BEARISH_SIGNAL
    elif bull_dom_cross:
        signal = BULLISH_SIGNAL
    elif bear_dom_cross:
        signal = BEARISH_SIGNAL

    # Trend determination:
    # - If both components are <= 1 or NaN, trend is neutral.
    # - Otherwise, whichever component is larger above 1 sets the trend direction.
    trend = NEUTRAL_TREND
    if valid(h_curr) and valid(l_curr):
        if max(h_curr, l_curr) > thr:
            trend = BULLISH_TREND if h_curr > l_curr else BEARISH_TREND
    elif valid(h_curr):
        trend = BULLISH_TREND if h_curr > thr else NEUTRAL_TREND
    elif valid(l_curr):
        trend = BEARISH_TREND if l_curr > thr else NEUTRAL_TREND

    return [signal, trend]