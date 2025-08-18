import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_2nd_order_gaussian_high_pass_filter_mtf_zones(series: pd.Series) -> list[str]:
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = series.iloc[-2]
    curr = series.iloc[-1]

    # Determine trend from current value
    if pd.notna(curr):
        if curr > 0:
            trend = BULLISH_TREND
        elif curr < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    else:
        trend = NEUTRAL_TREND

    # Signal from zero-line cross using last two bars
    signal = NO_SIGNAL
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0 and curr > 0:
            signal = BULLISH_SIGNAL
        elif prev >= 0 and curr < 0:
            signal = BEARISH_SIGNAL

    return [signal, trend]