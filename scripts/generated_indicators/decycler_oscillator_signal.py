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

def signal_decycler_oscillator(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Extract series safely
    deo = df["DEO"] if "DEO" in df.columns else pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    deo2 = df["DEO2"] if "DEO2" in df.columns else pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")

    # Signals: use zero-line cross of DEO2 (faster line) over the last two bars
    prev = deo2.iloc[-2]
    curr = deo2.iloc[-1]
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0 and curr > 0:
            tags.append(BULLISH_SIGNAL)
        elif prev >= 0 and curr < 0:
            tags.append(BEARISH_SIGNAL)
        else:
            tags.append(NO_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Trend: use current DEO (slower line). If NaN, fall back to DEO2. Zero implies neutral.
    curr_deo = deo.iloc[-1]
    trend_tag = NEUTRAL_TREND
    if pd.notna(curr_deo):
        if curr_deo > 0:
            trend_tag = BULLISH_TREND
        elif curr_deo < 0:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        curr_deo2 = deo2.iloc[-1]
        if pd.notna(curr_deo2):
            if curr_deo2 > 0:
                trend_tag = BULLISH_TREND
            elif curr_deo2 < 0:
                trend_tag = BEARISH_TREND
            else:
                trend_tag = NEUTRAL_TREND
        else:
            trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags