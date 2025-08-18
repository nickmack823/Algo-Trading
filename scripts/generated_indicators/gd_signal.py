import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_gd(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if "GD" not in df.columns or "EMA" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    ema = df["EMA"]
    gd = df["GD"]

    curr_ema = ema.iloc[-1]
    prev_ema = ema.iloc[-2]
    curr_gd = gd.iloc[-1]
    prev_gd = gd.iloc[-2]

    # Robust to NaNs
    if pd.isna(curr_ema) or pd.isna(prev_ema) or pd.isna(curr_gd) or pd.isna(prev_gd):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Cross detection using last two bars
    bullish_cross = (prev_ema <= prev_gd) and (curr_ema > curr_gd)
    bearish_cross = (prev_ema >= prev_gd) and (curr_ema < curr_gd)

    # Trend determination from current relationship
    if curr_ema > curr_gd:
        trend = BULLISH_TREND
    elif curr_ema < curr_gd:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    if bullish_cross:
        return [BULLISH_SIGNAL, trend]
    if bearish_cross:
        return [BEARISH_SIGNAL, trend]
    return [NO_SIGNAL, trend]