import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_vidya(series: pd.Series) -> list[str]:
    # Robustness: handle empty or too-short inputs
    if series is None or len(series) < 2:
        return [NO_SIGNAL, NEUTRAL_TREND]

    s = pd.to_numeric(series, errors='coerce')
    v_prev = s.iloc[-2]
    v_last = s.iloc[-1]

    # If either of the last two bars is NaN, be neutral
    if pd.isna(v_prev) or pd.isna(v_last):
        return [NO_SIGNAL, NEUTRAL_TREND]

    # Determine trend from the last two bars (slope of VIDYA)
    scale = max(abs(v_last), abs(v_prev), 1.0)
    eps = 1e-6 * scale

    if v_last > v_prev + eps:
        trend = BULLISH_TREND
    elif v_last < v_prev - eps:
        trend = BEARISH_TREND
    else:
        trend = NEUTRAL_TREND

    # Baseline role: emit trend; no entry signal without price-vs-baseline cross
    return [NO_SIGNAL, trend]