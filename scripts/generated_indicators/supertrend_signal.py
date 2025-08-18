import pandas as pd
from scripts.config import BULLISH_SIGNAL, BEARISH_SIGNAL, BULLISH_TREND, BEARISH_TREND, NEUTRAL_TREND, OVERBOUGHT, OVERSOLD, NO_SIGNAL, HIGH_VOLUME, LOW_VOLUME, HIGH_VOLATILITY, LOW_VOLATILITY, INCREASING_VOLATILITY, DECREASING_VOLATILITY, STABLE_VOLATILITY, INCONCLUSIVE

def signal_supertrend(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or 'Trend' not in df.columns:
        return [NO_SIGNAL, NEUTRAL_TREND]

    t = df['Trend']
    n = len(t)

    # Determine current trend tag
    curr = t.iloc[-1] if n >= 1 else float('nan')
    if pd.isna(curr):
        trend_tag = NEUTRAL_TREND
    elif curr > 0:
        trend_tag = BULLISH_TREND
    elif curr < 0:
        trend_tag = BEARISH_TREND
    else:
        trend_tag = NEUTRAL_TREND

    # Determine signal based on last two bars (flip detection)
    if n >= 2:
        prev = t.iloc[-2]
        if pd.notna(prev) and pd.notna(curr):
            if prev < 0 and curr > 0:
                signal = BULLISH_SIGNAL
            elif prev > 0 and curr < 0:
                signal = BEARISH_SIGNAL
            else:
                signal = NO_SIGNAL
        else:
            signal = NO_SIGNAL
    else:
        signal = NO_SIGNAL

    return [signal, trend_tag]