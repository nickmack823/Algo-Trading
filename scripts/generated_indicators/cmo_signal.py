import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_cmo(series: pd.Series) -> list[str]:
    tags: list[str] = []

    if series is None or len(series) == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    c0 = series.iloc[-1] if len(series) >= 1 else float("nan")
    c1 = series.iloc[-2] if len(series) >= 2 else float("nan")

    # Overbought / Oversold tags based on current value
    if pd.notna(c0):
        if c0 >= 50:
            tags.append(OVERBOUGHT)
        elif c0 <= -50:
            tags.append(OVERSOLD)

    # Signal generation using last two bars
    signaled = False
    if pd.notna(c0) and pd.notna(c1):
        bullish_cross_zero = c1 <= 0 and c0 > 0
        bearish_cross_zero = c1 >= 0 and c0 < 0
        bullish_exit_oversold = c1 <= -50 and c0 > -50
        bearish_exit_overbought = c1 >= 50 and c0 < 50

        if bullish_cross_zero or bullish_exit_oversold:
            tags.append(BULLISH_SIGNAL)
            signaled = True
        elif bearish_cross_zero or bearish_exit_overbought:
            tags.append(BEARISH_SIGNAL)
            signaled = True

    if not signaled:
        tags.append(NO_SIGNAL)

    # Trend tag (final tag)
    if pd.notna(c0):
        if c0 > 0:
            trend = BULLISH_TREND
        elif c0 < 0:
            trend = BEARISH_TREND
        else:
            trend = NEUTRAL_TREND
    else:
        trend = NEUTRAL_TREND

    tags.append(trend)
    return tags