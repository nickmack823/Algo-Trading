import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_frama_indicator(series: pd.Series) -> list[str]:
    # Ensure numeric and handle NaNs gracefully
    if series is None or len(series) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.to_numeric(series, errors="coerce")
    n = len(s)

    # Determine trend based on FRAMA slope (last two bars)
    trend = NEUTRAL_TREND
    s0 = s.iloc[-1] if n >= 1 else float("nan")
    s1 = s.iloc[-2] if n >= 2 else float("nan")
    if pd.notna(s0) and pd.notna(s1):
        if s0 > s1:
            trend = BULLISH_TREND
        elif s0 < s1:
            trend = BEARISH_TREND

    # Determine signal based on slope crossing zero (uses last two bars' slope comparison)
    signal = NO_SIGNAL
    if n >= 3:
        s2 = s.iloc[-3]
        if pd.notna(s0) and pd.notna(s1) and pd.notna(s2):
            diff_prev = s1 - s2
            diff_curr = s0 - s1
            if diff_prev <= 0 and diff_curr > 0:
                signal = BULLISH_SIGNAL
            elif diff_prev >= 0 and diff_curr < 0:
                signal = BEARISH_SIGNAL

    return [signal, trend]