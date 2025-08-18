import pandas as pd
from scripts.config import (
    BULLISH_SIGNAL,
    BEARISH_SIGNAL,
    BULLISH_TREND,
    BEARISH_TREND,
    NEUTRAL_TREND,
    NO_SIGNAL,
)


def signal_trend_akkam(df: pd.DataFrame) -> list[str]:
    # Expecting DataFrame with at least 'TrStop' column
    if not isinstance(df, pd.DataFrame) or 'TrStop' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.to_numeric(df['TrStop'], errors='coerce')
    n = len(s)

    if n < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    t_prev = s.iloc[-2]
    t_curr = s.iloc[-1]

    if pd.isna(t_prev) or pd.isna(t_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Current slope (line color proxy)
    delta_curr = t_curr - t_prev
    curr_slope = 1 if delta_curr > 0 else (-1 if delta_curr < 0 else 0)

    # Previous slope to detect "turns green/red" events
    prev_slope = None
    if n >= 3:
        t_prev2 = s.iloc[-3]
        if pd.notna(t_prev2):
            delta_prev = t_prev - t_prev2
            prev_slope = 1 if delta_prev > 0 else (-1 if delta_prev < 0 else 0)

    # Determine signal on slope change
    signal = NO_SIGNAL
    if prev_slope is not None:
        if curr_slope > 0 and prev_slope <= 0:
            signal = BULLISH_SIGNAL
        elif curr_slope < 0 and prev_slope >= 0:
            signal = BEARISH_SIGNAL

    # Trend tag from current slope
    if curr_slope > 0:
        trend = BULLISH_TREND
    elif curr_slope < 0:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    return [signal, trend]