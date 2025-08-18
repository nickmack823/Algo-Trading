import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, INCONCLUSIVE

def signal_metro_fixed(df: pd.DataFrame) -> list[str]:
    tags: list[str] = []

    # Basic validation
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    required_cols = {"RSI", "StepRSI_slow"}
    if not required_cols.issubset(df.columns):
        return [INCONCLUSIVE, NEUTRAL_TREND]

    rsi = df["RSI"]
    slow = df["StepRSI_slow"]

    # Last two bars
    rsi_prev = rsi.iloc[-2]
    rsi_curr = rsi.iloc[-1]
    slow_prev = slow.iloc[-2]
    slow_curr = slow.iloc[-1]

    # Robustness to NaNs
    if pd.isna(rsi_prev) or pd.isna(rsi_curr) or pd.isna(slow_prev) or pd.isna(slow_curr):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Signal: crossover of RSI above/below StepRSI_slow (FRAMA-style interpretation)
    bullish_cross = (rsi_prev <= slow_prev) and (rsi_curr > slow_curr)
    bearish_cross = (rsi_prev >= slow_prev) and (rsi_curr < slow_curr)

    if bullish_cross:
        tags.append(BULLISH_SIGNAL)
    elif bearish_cross:
        tags.append(BEARISH_SIGNAL)
    else:
        tags.append(NO_SIGNAL)

    # Overbought/Oversold context on current bar
    if rsi_curr >= 70.0:
        tags.append(OVERBOUGHT)
    elif rsi_curr <= 30.0:
        tags.append(OVERSOLD)

    # Trend: current RSI relative to StepRSI_slow
    if rsi_curr > slow_curr:
        trend_tag = BULLISH_TREND
    elif rsi_curr < slow_curr:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    tags.append(trend_tag)
    return tags