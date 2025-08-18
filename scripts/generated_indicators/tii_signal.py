import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_tii(series: pd.Series) -> list[str]:
    up_th = 60.0
    down_th = 40.0

    # Handle insufficient data
    if series is None or len(series) < 1:
        return [NO_SIGNAL, NEUTRAL_TREND]
    last = series.iloc[-1]

    # Determine trend from the latest value
    if pd.isna(last):
        trend_tag = NEUTRAL_TREND
    elif last > up_th:
        trend_tag = BULLISH_TREND
    elif last < down_th:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Need two bars for cross signals
    if len(series) < 2 or pd.isna(last) or pd.isna(series.iloc[-2]):
        return [NO_SIGNAL, trend_tag]

    prev = series.iloc[-2]

    bullish_cross = (prev <= up_th) and (last > up_th)
    bearish_cross = (prev >= down_th) and (last < down_th)

    if bullish_cross:
        return [BULLISH_SIGNAL, trend_tag]
    if bearish_cross:
        return [BEARISH_SIGNAL, trend_tag]

    return [NO_SIGNAL, trend_tag]