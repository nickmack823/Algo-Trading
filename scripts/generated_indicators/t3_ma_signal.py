import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_t3_ma(series: pd.Series) -> list[str]:
    # Robust to NaNs and short series
    if series is None or not isinstance(series, pd.Series):
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = series.dropna()
    if len(s) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    prev_val = s.iloc[-2]
    curr_val = s.iloc[-1]

    # Determine trend from last two bars of T3
    if pd.isna(prev_val) or pd.isna(curr_val):
        trend_tag = NEUTRAL_TREND
    elif curr_val > prev_val:
        trend_tag = BULLISH_TREND
    elif curr_val < prev_val:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Without price series, we cannot confirm price-vs-baseline cross; emit NO_SIGNAL
    return [NO_SIGNAL, trend_tag]