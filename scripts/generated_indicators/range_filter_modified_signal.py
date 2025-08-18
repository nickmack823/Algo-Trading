import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, NO_SIGNAL

def signal_range_filter_modified(df: pd.DataFrame) -> list[str]:
    # Robust defaults
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] < 2 or 'LineCenter' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    lc = df['LineCenter']
    if lc.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = lc.iloc[-2]
    curr = lc.iloc[-1]

    signal = NO_SIGNAL
    trend = NEUTRAL_TREND

    if pd.notna(prev) and pd.notna(curr):
        if curr > prev:
            signal = BULLISH_SIGNAL
            trend = BULLISH_TREND
        elif curr < prev:
            signal = BEARISH_SIGNAL
            trend = BEARISH_TREND
        else:
            signal = NO_SIGNAL
            trend = NEUTRAL_TREND
    else:
        signal = NO_SIGNAL
        trend = NEUTRAL_TREND

    return [signal, trend]