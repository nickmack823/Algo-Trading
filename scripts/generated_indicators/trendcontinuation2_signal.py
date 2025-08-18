import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_trendcontinuation2(df: pd.DataFrame) -> list[str]:
    # Validate input
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    pos = df.get("TrendContinuation2_Pos")
    neg = df.get("TrendContinuation2_Neg")
    if pos is None or neg is None:
        return [NO_SIGNAL, NEUTRAL_TREND]

    reading = pos - neg

    prev = reading.iloc[-2]
    curr = reading.iloc[-1]

    # Determine trend tag from current value
    if pd.notna(curr) and curr > 0:
        trend_tag = BULLISH_TREND
    elif pd.notna(curr) and curr < 0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Signal on zero-cross using last two bars
    signal_tag = NO_SIGNAL
    if pd.notna(prev) and pd.notna(curr):
        if prev <= 0 and curr > 0:
            signal_tag = BULLISH_SIGNAL
        elif prev >= 0 and curr < 0:
            signal_tag = BEARISH_SIGNAL

    return [signal_tag, trend_tag]