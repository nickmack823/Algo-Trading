import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_ehlers_two_pole_super_smoother_filter(series: pd.Series) -> list[str]:
    # Robust to empty/short/NaN series
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = series.iloc[-2]
    curr = series.iloc[-1]

    if pd.isna(prev) or pd.isna(curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Trend from last two bars (slope of the baseline)
    if curr > prev:
        trend = BULLISH_TREND
    elif curr < prev:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    # Without price context, do not emit entry signals; provide trend only
    return [NO_SIGNAL, trend]