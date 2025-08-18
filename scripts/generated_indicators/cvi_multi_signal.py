import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_cvi_multi(series: pd.Series) -> list[str]:
    # Expecting a CVI Series where sign indicates price relative to baseline
    try:
        s = series.astype(float)
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if s.shape[0] < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = s.iloc[-2]
    curr = s.iloc[-1]

    if pd.isna(prev) or pd.isna(curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    tags: list[str] = []

    # Cross of zero line implies baseline cross
    if prev <= 0.0 and curr > 0.0:
        tags.append(BULLISH_SIGNAL)
    elif prev >= 0.0 and curr < 0.0:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Optional OB/OS based on normalized magnitude (ATR units)
    ob_level = 2.0
    if curr >= ob_level:
        tags.append(OVERBOUGHT)
    elif curr <= -ob_level:
        tags.append(OVERSOLD)

    # Trend tag (exactly one, appended last)
    if curr > 0.0:
        tags.append(BULLISH_TREND)
    elif curr < 0.0:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags