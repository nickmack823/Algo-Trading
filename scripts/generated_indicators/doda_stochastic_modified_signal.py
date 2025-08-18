import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_doda_stochastic_modified(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Validate inputs
    if df is None or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Extract lines
    try:
        k = df["DodaStoch"]
        d = df["DodaSignal"]
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    if len(k) < 2 or len(d) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    k_prev, k_curr = k.iloc[-2], k.iloc[-1]
    d_prev, d_curr = d.iloc[-2], d.iloc[-1]

    if pd.isna(k_prev) or pd.isna(k_curr) or pd.isna(d_prev) or pd.isna(d_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Overbought/Oversold states based on current values
    if (k_curr <= 20) or (d_curr <= 20):
        tags.append(OVERSOLD)
    elif (k_curr >= 80) or (d_curr >= 80):
        tags.append(OVERBOUGHT)

    # Cross detections using last two bars
    bullish_cross = (k_prev <= d_prev) and (k_curr > d_curr)
    bearish_cross = (k_prev >= d_prev) and (k_curr < d_curr)

    # Signal only if cross occurs in appropriate territory (~20 for long, ~80 for short)
    if bullish_cross and ((k_curr <= 20) or (d_curr <= 20) or (k_prev <= 20) or (d_prev <= 20)):
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross and ((k_curr >= 80) or (d_curr >= 80) or (k_prev >= 80) or (d_prev >= 80)):
        tags.append(BEARISH_SIGNAL)

    # If no explicit signal, mark NO_SIGNAL
    if BULLISH_SIGNAL not in tags and BEARISH_SIGNAL not in tags:
        tags.append(NO_SIGNAL)

    # Trend determination from current relation of K vs D
    if k_curr > d_curr:
        trend_tag = BULLISH_TREND
    elif k_curr < d_curr:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags