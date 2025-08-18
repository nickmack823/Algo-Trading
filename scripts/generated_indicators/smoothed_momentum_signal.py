import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_smoothed_momentum(df: pd.DataFrame) -> list[str]:
    # Expecting DataFrame with column 'SM' from SmoothedMomentum(...)
    if df is None or df.empty or 'SM' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = df['SM'].astype(float)
    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev = s.iloc[-2]
    curr = s.iloc[-1]

    if pd.isna(prev) or pd.isna(curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    baseline = 100.0  # Equivalent to "zero line" for this momentum measure

    tags: list[str] = []

    # Signals: cross of baseline using the last two bars
    if prev <= baseline and curr > baseline:
        tags.append(BULLISH_SIGNAL)
    elif prev >= baseline and curr < baseline:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend tag: based on current position relative to baseline
    if curr > baseline:
        tags.append(BULLISH_TREND)
    elif curr < baseline:
        tags.append(BEARISH_TREND)
    else:
        tags.append(NEUTRAL_TREND)

    return tags