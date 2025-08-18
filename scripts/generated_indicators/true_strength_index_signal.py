import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_true_strength_index(series: pd.Series) -> list[str]:
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    curr = series.iloc[-1]
    prev = series.iloc[-2]

    if pd.isna(curr) or pd.isna(prev):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tags: list[str] = []

    bullish_cross = (prev <= 0) and (curr > 0)
    bearish_cross = (prev >= 0) and (curr < 0)

    if bullish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross:
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