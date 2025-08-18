import pandas as pd
from scripts.config import (
    BULLISH_SIGNAL,
    BEARISH_SIGNAL,
    BULLISH_TREND,
    BEARISH_TREND,
    NEUTRAL_TREND,
    NO_SIGNAL,
)

def signal_trend_lord_nrp_indicator(df: pd.DataFrame) -> list[str]:
    # Validate input
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if "Buy" not in df.columns or "Sell" not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    b_prev = df["Buy"].iloc[-2]
    b_curr = df["Buy"].iloc[-1]
    s_prev = df["Sell"].iloc[-2]
    s_curr = df["Sell"].iloc[-1]

    # Signal: crossover between Buy (ma1/slot) and Sell (ma2) on last two bars
    signal_tag = NO_SIGNAL
    if pd.notna(b_prev) and pd.notna(b_curr) and pd.notna(s_prev) and pd.notna(s_curr):
        crossed_up = b_prev <= s_prev and b_curr > s_curr
        crossed_down = b_prev >= s_prev and b_curr < s_curr
        if crossed_up:
            signal_tag = BULLISH_SIGNAL
        elif crossed_down:
            signal_tag = BEARISH_SIGNAL

    # Trend: slope of Sell (ma2) on last two bars
    if pd.notna(s_prev) and pd.notna(s_curr):
        if s_curr > s_prev:
            trend_tag = BULLISH_TREND
        elif s_curr < s_prev:
            trend_tag = BEARISH_TREND
        else:
            trend_tag = NEUTRAL_TREND
    else:
        trend_tag = NEUTRAL_TREND

    return [signal_tag, trend_tag]