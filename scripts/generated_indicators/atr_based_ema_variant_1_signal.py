import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_atr_based_ema_variant_1(df: pd.DataFrame) -> list[str]:
    # Expecting columns: ['EMA_ATR_var1', 'EMA_Equivalent']
    try:
        ema = pd.to_numeric(df['EMA_ATR_var1'], errors='coerce')
    except Exception:
        return [NO_SIGNAL, NEUTRAL_TREND]

    ema_nonan = ema.dropna()
    n = len(ema_nonan)

    if n == 0:
        return [NO_SIGNAL, NEUTRAL_TREND]
    if n == 1:
        return [NO_SIGNAL, NEUTRAL_TREND]

    last = float(ema_nonan.iloc[-1])
    prev = float(ema_nonan.iloc[-2])

    # Trend based on slope of baseline over the last two bars
    if last > prev:
        trend_tag = BULLISH_TREND
    elif last < prev:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Signal on slope direction change (uses last three valid points if available)
    signal_tag = NO_SIGNAL
    if n >= 3:
        prev2 = float(ema_nonan.iloc[-3])
        delta_prev = prev - prev2
        delta_now = last - prev

        sign_prev = 1 if delta_prev > 0 else (-1 if delta_prev < 0 else 0)
        sign_now = 1 if delta_now > 0 else (-1 if delta_now < 0 else 0)

        if sign_now != 0 and sign_now != sign_prev:
            signal_tag = BULLISH_SIGNAL if sign_now > 0 else BEARISH_SIGNAL

    return [signal_tag, trend_tag]