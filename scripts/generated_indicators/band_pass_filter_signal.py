import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_band_pass_filter(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or 'BP' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    bp = df['BP']
    n = len(bp)

    # Determine trend from latest available value (prefer current, then previous)
    def trend_from_value(v):
        if pd.isna(v):
            return NEUTRAL_TREND
        if v > 0:
            return BULLISH_TREND
        if v < 0:
            return BEARISH_TREND
        return NEUTRAL_TREND

    if n < 2:
        last_val = bp.iloc[-1] if n == 1 else float('nan')
        return [NO_SIGNAL, trend_from_value(last_val)]

    prev = bp.iloc[-2]
    curr = bp.iloc[-1]

    # Signal logic: zero-cross over the last two bars
    signal = NO_SIGNAL
    if not pd.isna(prev) and not pd.isna(curr):
        if prev <= 0 and curr > 0:
            signal = BULLISH_SIGNAL
        elif prev >= 0 and curr < 0:
            signal = BEARISH_SIGNAL

    # Trend tag based on latest available data (current preferred, else previous)
    if not pd.isna(curr):
        trend_tag = trend_from_value(curr)
    elif not pd.isna(prev):
        trend_tag = trend_from_value(prev)
    else:
        trend_tag = NEUTRAL_TREND

    return [signal, trend_tag]