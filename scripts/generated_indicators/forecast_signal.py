import pandas as pd
from scripts.config import (
    BULLISH_SIGNAL,
    BEARISH_SIGNAL,
    BULLISH_TREND,
    BEARISH_TREND,
    NEUTRAL_TREND,
    OVERBOUGHT,
    OVERSOLD,
    NO_SIGNAL,
    HIGH_VOLUME,
    LOW_VOLUME,
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    INCREASING_VOLATILITY,
    DECREASING_VOLATILITY,
    STABLE_VOLATILITY,
    INCONCLUSIVE,
)

def signal_forecast(series: pd.Series) -> list[str]:
    tags: list[str] = []

    n = len(series) if series is not None else 0
    prev = float(series.iloc[-2]) if n >= 2 else float("nan")
    curr = float(series.iloc[-1]) if n >= 1 else float("nan")

    cross_up = (prev <= 0) and (curr > 0)
    cross_down = (prev >= 0) and (curr < 0)

    if cross_up:
        tags.append(BULLISH_SIGNAL)
    elif cross_down:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    if curr > 0:
        tags.append(BULLISH_TREND)
    elif curr < 0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags