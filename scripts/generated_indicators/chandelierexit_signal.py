import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_chandelierexit(df: pd.DataFrame) -> list[str]:
    # Validate input and coerce to numeric
    try:
        long_line = pd.to_numeric(df["Chandelier_Long"], errors="coerce")
        short_line = pd.to_numeric(df["Chandelier_Short"], errors="coerce")
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    n = len(df)
    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Current state
    last_long = long_line.iloc[-1] if n >= 1 else float("nan")
    last_short = short_line.iloc[-1] if n >= 1 else float("nan")
    curr_long_active = pd.notna(last_long) and pd.isna(last_short)
    curr_short_active = pd.notna(last_short) and pd.isna(last_long)

    # Trend tag
    if curr_long_active:
        trend = BULLISH_TREND
    elif curr_short_active:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    # Signal based on flip between last two bars
    signal = NO_SIGNAL
    if n >= 2:
        prev_long = long_line.iloc[-2]
        prev_short = short_line.iloc[-2]
        prev_long_active = pd.notna(prev_long) and pd.isna(prev_short)
        prev_short_active = pd.notna(prev_short) and pd.isna(prev_long)

        if prev_long_active and curr_short_active:
            signal = BEARISH_SIGNAL  # Exit longs
        elif prev_short_active and curr_long_active:
            signal = BULLISH_SIGNAL  # Exit shorts

    return [signal, trend]