import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_geomin_ma(series: pd.Series) -> list[str]:
    # Robustness: need at least 3 values to detect a turn (baseline turning up/down)
    if series is None or len(series) < 3:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s1 = series.iloc[-1]
    s2 = series.iloc[-2]
    s3 = series.iloc[-3]

    # Handle NaNs in the last three bars
    if pd.isna(s1) or pd.isna(s2) or pd.isna(s3):
        return [NO_SIGNAL, NEUTRAL_TREND]

    d_now = s1 - s2
    d_prev = s2 - s3

    tags: list[str] = []

    # Signal on baseline turning points
    if d_prev <= 0 and d_now > 0:
        tags.append(BULLISH_SIGNAL)
    elif d_prev >= 0 and d_now < 0:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend from current slope
    if d_now > 0:
        tags.append(BULLISH_TREND)
    elif d_now < 0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags